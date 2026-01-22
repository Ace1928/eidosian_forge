import copy
import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
import onnxruntime as ort
from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..utils import check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.normalized_config import NormalizedConfigManager
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .base import ORTDecoderForSeq2Seq, ORTEncoder
from .constants import (
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
from huggingface_hub.utils import EntryNotFoundError
class ORTModelForConditionalGeneration(ORTModel, ABC):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.

    Important attributes:
        config ([`PretrainedConfig`]):
            Instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        use_io_binding (`Optional[bool]`, defaults to `None`):
            Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to `True`
            if the device is CUDA, otherwise defaults to `False`.
        use_cache (`bool`):
            Whether or not past key/values cache should be used. It is determined by whether an InferenceSession for
            that was provided or not.
        providers (`List[str`]):
            The list of execution providers the model is running on.
        encoder (`ORTEncoder`):
            The encoder model.
        decoder (`ORTDecoderForSeq2Seq`):
            The decoder model.
        decoder_with_past (`Optional[ORTDecoderForSeq2Seq]`):
            The decoder model handling the past key/values if `use_cache=True`, else `None`.

    Other attributes:
        encoder_file_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_ENCODER_NAME`):
            The name of the ONNX file containing the encoder part of the model.
        decoder_file_name (`str`,  defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
            The name of the ONNX file containing the decoder part of the model.
        decoder_file_with_past_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
            The name of the ONNX file containing the decoder with past key/values part of the model.
        model_save_dir (`str`, defaults to `""`):
            The directory under which the model exported to ONNX was saved.

    """
    base_model_prefix = 'onnx_model'

    def __init__(self, encoder_session: ort.InferenceSession, decoder_session: ort.InferenceSession, config: 'PretrainedConfig', onnx_paths: List[str], decoder_with_past_session: Optional[ort.InferenceSession]=None, use_cache: bool=True, use_io_binding: Optional[bool]=None, model_save_dir: Optional[Union[str, Path, TemporaryDirectory]]=None, preprocessors: Optional[List]=None, generation_config: Optional[GenerationConfig]=None, **kwargs):
        """
        Args:
            encoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the encoder.
            decoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            config ([`PretrainedConfig`]):
                `config` is an instance of the configuration associated to the model. Initializing with a config file
                does not load the weights associated with the model, only the configuration.
            onnx_paths (`List[str]`):
                Path to ONNX files associated with the model.
            decoder_with_past_session (`Optional[ort.InferenceSession]`, *optional*):
                The ONNX Runtime inference session associated to the decoder with past key values.
            use_io_binding (`bool`, *optional*, defaults to `None`):
                Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`str`, *optional*, defaults to `""`):
                The directory under which the model exported to ONNX was saved.
            preprocessors (`Optional[List]`, defaults to `None`):
                The list of the preprocessors (tokenizer, processor, feature_extractor) to save alongside the ORTModel.
            generation_config (`Optional[GenerationConfig]`, defaults to `None`):
                The generation configuration used by default when calling `generate()`.
                Refer to https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.
        """

        def show_deprecated_argument(arg_name):
            if kwargs.pop(arg_name, None) is not None:
                logger.warning(f'The {arg_name} argument to create an {self.__class__.__name__} is deprecated, and not used anymore.')
        show_deprecated_argument('last_encoder_model_name')
        show_deprecated_argument('last_decoder_model_name')
        show_deprecated_argument('last_decoder_with_past_model_name')
        if kwargs:
            raise ValueError(f'{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments.')
        ABC.__init__(self)
        if use_io_binding is None:
            if decoder_session.get_providers()[0] == 'CUDAExecutionProvider':
                use_io_binding = True
            else:
                use_io_binding = False
        self.shared_attributes_init(encoder_session, use_io_binding=use_io_binding, model_save_dir=model_save_dir, preprocessors=preprocessors)
        self.config = config
        self.name_or_path = config.name_or_path
        self.onnx_paths = onnx_paths
        self.use_cache = use_cache
        if use_cache is True:
            use_merged = 'use_cache_branch' in [inp.name for inp in decoder_session.get_inputs()]
            if use_merged is True and decoder_with_past_session is not None:
                raise ValueError('Detected a merged decoder, but decoder_with_past_session was provided.Please only set decoder_session, or provide a non-merged decoder_session.')
            if use_cache is True and use_merged is False and (decoder_with_past_session is None):
                raise ValueError('The parameter use_cache was set as True, but neither decoder_with_past_session was passed nor a use_cache branch can be found in the decoder_session. Please pass a decoder_with_past_session or set use_cache=False.')
        else:
            use_merged = False
            if decoder_with_past_session is not None:
                raise ValueError('The parameter decoder_with_past_session was passed, although use_cache is False.Please pass use_cache=True for decoder_with_past_session to be used.')
        if use_cache is False and use_io_binding is True:
            raise ValueError('When using CUDAExecutionProvider, the parameters combination use_cache=False, use_io_binding=True is not supported. Please either pass use_cache=True, use_io_binding=True (default), or use_cache=False, use_io_binding=False.')
        self.use_merged = use_merged
        self.encoder = self._initialize_encoder(encoder_session)
        self.encoder_model_path = Path(encoder_session._model_path)
        self.encoder_model_name = self.encoder_model_path.name
        self.decoder = ORTDecoderForSeq2Seq(decoder_session, self)
        self.decoder_model_path = Path(decoder_session._model_path)
        self.decoder_model_name = self.decoder_model_path.name
        self.decoder_with_past = None
        self.decoder_with_past_model_path = None
        self.decoder_with_past_model_name = None
        if self.use_cache is True and self.use_merged is False:
            self.decoder_with_past = ORTDecoderForSeq2Seq(decoder_with_past_session, self)
            self.decoder_with_past_model_path = Path(decoder_with_past_session._model_path)
            self.decoder_with_past_model_name = self.decoder_with_past_model_path.name
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

    @abstractmethod
    def _initialize_encoder(self, session: ort.InferenceSession) -> ORTEncoder:
        pass

    @staticmethod
    def load_model(encoder_path: Union[str, Path], decoder_path: Union[str, Path], decoder_with_past_path: Optional[Union[str, Path]]=None, provider: str='CPUExecutionProvider', session_options: Optional[ort.SessionOptions]=None, provider_options: Optional[Dict]=None):
        """
        Creates an instance of [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForConditionalGeneration`].
        Three inference sessions will be created for respectively the encoder, decoder and decoder with past key values
        models. The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            encoder_path (`Union[str, Path]`):
                The path of the encoder ONNX model.
            decoder_path (`Union[str, Path]`):
                The path of the decoder ONNX model.
            decoder_with_past_path (`Optional[Union[str, Path]]`, *optional*):
                The path of the decoder with past key values ONNX model.
            provider (`str`, *optional*, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[ort.SessionOptions]`, *optional*),:
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Optional[Dict]`, *optional*):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        encoder_session = ORTModel.load_model(encoder_path, provider, session_options, provider_options)
        decoder_session = ORTModel.load_model(decoder_path, provider, session_options, provider_options)
        decoder_with_past_session = None
        if decoder_with_past_path is not None:
            decoder_with_past_session = ORTModel.load_model(decoder_with_past_path, provider, session_options, provider_options)
        return (encoder_session, decoder_session, decoder_with_past_session)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model encoder, decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForSeq2SeqLM.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path`]):
                The directory where to save the model files.
        """
        save_directory = Path(save_directory)
        src_paths = [Path(path) for path in self.onnx_paths]
        dst_paths = [save_directory / path.name for path in src_paths]
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)
        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)
        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: 'PretrainedConfig', use_auth_token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, force_download: bool=False, cache_dir: Optional[str]=None, encoder_file_name: str=ONNX_ENCODER_NAME, decoder_file_name: str=ONNX_DECODER_NAME, decoder_with_past_file_name: str=ONNX_DECODER_WITH_PAST_NAME, subfolder: str='', local_files_only: bool=False, use_cache: bool=True, use_merged: Optional[bool]=None, provider: str='CPUExecutionProvider', session_options: Optional[ort.SessionOptions]=None, provider_options: Optional[Dict[str, Any]]=None, use_io_binding: Optional[bool]=None, model_save_dir: Optional[Union[str, Path, TemporaryDirectory]]=None, **kwargs):
        model_path = Path(model_id)
        if use_cache is False:
            if use_merged is True:
                raise ValueError('The parameters combination use_cache=False, use_merged=True is not supported. To use a merged decoder, past key values must be used.')
            use_merged = False
        decoder_merged_path = None
        if use_merged is not False:
            try:
                decoder_merged_path = ORTModelForConditionalGeneration.infer_onnx_filename(model_id, [DECODER_MERGED_ONNX_FILE_PATTERN], argument_name=None, subfolder=subfolder, use_auth_token=use_auth_token, revision=revision)
                use_merged = True
                decoder_path = decoder_merged_path
            except FileNotFoundError as e:
                if use_merged is True:
                    raise FileNotFoundError(f'The parameter `use_merged=True` was passed to ORTModelForCausalLM.from_pretrained() but no ONNX file for a merged decoder could be found in {str(Path(model_id, subfolder))}, with the error: {e}')
                use_merged = False
        decoder_without_past_path = None
        decoder_with_past_path = None
        if use_merged is False:
            if not validate_file_exists(model_id, decoder_file_name, subfolder=subfolder, revision=revision):
                decoder_without_past_path = ORTModelForConditionalGeneration.infer_onnx_filename(model_id, [DECODER_ONNX_FILE_PATTERN], 'decoder_file_name', subfolder=subfolder, use_auth_token=use_auth_token, revision=revision)
            else:
                decoder_without_past_path = model_path / subfolder / decoder_file_name
            decoder_path = decoder_without_past_path
            decoder_regular_onnx_filenames = ORTModelForConditionalGeneration._generate_regular_names_for_filename(ONNX_DECODER_NAME)
            if decoder_path.name not in decoder_regular_onnx_filenames:
                logger.warning(f'The ONNX file {decoder_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_regular_onnx_filenames}, the {cls.__name__} might not behave as expected.')
            if use_cache is True and use_merged is False:
                if not validate_file_exists(model_id, decoder_with_past_file_name, subfolder=subfolder, revision=revision):
                    try:
                        decoder_with_past_path = ORTModelForConditionalGeneration.infer_onnx_filename(model_id, [DECODER_WITH_PAST_ONNX_FILE_PATTERN], 'decoder_with_past_file_name', subfolder=subfolder, use_auth_token=use_auth_token, revision=revision)
                    except FileNotFoundError as e:
                        raise FileNotFoundError(f'The parameter `use_cache=True` was passed to ORTModelForCausalLM.from_pretrained() but no ONNX file using past key values could be found in {str(Path(model_id, subfolder))}, with the error: {e}')
                else:
                    decoder_with_past_path = model_path / subfolder / decoder_with_past_file_name
                decoder_path = decoder_without_past_path
                decoder_with_past_regular_onnx_filenames = ORTModelForConditionalGeneration._generate_regular_names_for_filename(ONNX_DECODER_WITH_PAST_NAME)
                if decoder_with_past_path.name not in decoder_with_past_regular_onnx_filenames:
                    logger.warning(f'The ONNX file {decoder_with_past_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_with_past_regular_onnx_filenames}, the {cls.__name__} might not behave as expected.')
        if not validate_file_exists(model_id, encoder_file_name, subfolder=subfolder, revision=revision):
            encoder_path = ORTModelForConditionalGeneration.infer_onnx_filename(model_id, [ENCODER_ONNX_FILE_PATTERN], 'encoder_file_name', subfolder=subfolder, use_auth_token=use_auth_token, revision=revision)
        else:
            encoder_path = model_path / subfolder / encoder_file_name
        encoder_regular_onnx_filenames = ORTModelForConditionalGeneration._generate_regular_names_for_filename(ONNX_ENCODER_NAME)
        if encoder_path.name not in encoder_regular_onnx_filenames:
            logger.warning(f'The ONNX file {encoder_path.name} is not a regular name used in optimum.onnxruntime, the ORTModelForConditionalGeneration might not behave as expected.')
        preprocessors = None
        if model_path.is_dir():
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            attribute_name_to_filename = {'last_encoder_model_name': encoder_path.name, 'last_decoder_model_name': decoder_path.name if use_merged is False else None, 'last_decoder_with_past_model_name': decoder_with_past_path.name if use_merged is False and use_cache is True else None, 'last_decoder_merged_name': decoder_merged_path.name if use_merged is True else None}
            paths = {}
            for attr_name, filename in attribute_name_to_filename.items():
                if filename is None:
                    continue
                model_cache_path = hf_hub_download(repo_id=model_id, subfolder=subfolder, filename=filename, use_auth_token=use_auth_token, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only)
                try:
                    hf_hub_download(repo_id=model_id, subfolder=subfolder, filename=filename + '_data', use_auth_token=use_auth_token, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only)
                except EntryNotFoundError:
                    pass
                paths[attr_name] = Path(model_cache_path).name
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)
            if use_merged is True:
                decoder_path = new_model_save_dir / paths['last_decoder_merged_name']
                decoder_merged_path = new_model_save_dir / paths['last_decoder_merged_name']
            else:
                decoder_path = new_model_save_dir / paths['last_decoder_model_name']
                decoder_without_past_path = new_model_save_dir / paths['last_decoder_model_name']
                if use_cache is True:
                    decoder_with_past_path = new_model_save_dir / paths['last_decoder_with_past_model_name']
            encoder_path = new_model_save_dir / paths['last_encoder_model_name']
        ort_inference_sessions = cls.load_model(encoder_path=encoder_path, decoder_path=decoder_path, decoder_with_past_path=None if use_merged is True or use_cache is False else decoder_with_past_path, provider=provider, session_options=session_options, provider_options=provider_options)
        if model_save_dir is None:
            model_save_dir = new_model_save_dir
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_id, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only, use_auth_token=use_auth_token, revision=revision, subfolder=subfolder)
        except OSError:
            logger.info('Generation config file not found, using a generation config created from the model config.')
        onnx_paths = [encoder_path]
        if use_merged is False:
            onnx_paths.append(decoder_without_past_path)
            if use_cache is True:
                onnx_paths.append(decoder_with_past_path)
        else:
            onnx_paths.append(decoder_merged_path)
        return cls(*ort_inference_sessions[:2], config, onnx_paths=onnx_paths, use_cache=use_cache, decoder_with_past_session=ort_inference_sessions[2], use_io_binding=use_io_binding, model_save_dir=model_save_dir, preprocessors=preprocessors, generation_config=generation_config)

    @classmethod
    def _from_transformers(cls, model_id: str, config: 'PretrainedConfig', use_auth_token: Optional[Union[bool, str]]=None, revision: str='main', force_download: bool=True, cache_dir: Optional[str]=None, subfolder: str='', local_files_only: bool=False, trust_remote_code: bool=False, use_cache: bool=True, use_merged: bool=False, provider: str='CPUExecutionProvider', session_options: Optional[ort.SessionOptions]=None, provider_options: Optional[Dict[str, Any]]=None, use_io_binding: Optional[bool]=None, task: Optional[str]=None) -> 'ORTModelForConditionalGeneration':
        if use_cache is False and use_merged is True:
            raise ValueError('The incompatible arguments use_cache=False, use_merged=True were passed to ORTModelForConditionalGeneration.from_pretrained(). Please pass either use_cache=False, use_merged=False to disable past key value caching, or use_cache=True, use_merged=False to disable the merging of the decoder not using / using past key and value.')
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)
            if use_cache is True:
                task = task + '-with-past'
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        main_export(model_name_or_path=model_id, output=save_dir_path, task=task, do_validation=False, no_post_process=not use_merged, subfolder=subfolder, revision=revision, cache_dir=cache_dir, use_auth_token=use_auth_token, local_files_only=local_files_only, force_download=force_download, trust_remote_code=trust_remote_code)
        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)
        return cls._from_pretrained(save_dir_path, config, use_cache=use_cache, use_merged=use_merged, provider=provider, session_options=session_options, provider_options=provider_options, use_io_binding=use_io_binding, model_save_dir=save_dir)

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """
        device, provider_options = parse_device(device)
        if device.type == 'cuda' and self.providers[0] == 'TensorrtExecutionProvider':
            return self
        provider = get_provider_for_device(device)
        validate_provider_availability(provider)
        self.device = device
        self.encoder.session.set_providers([provider], provider_options=[provider_options])
        self.decoder.session.set_providers([provider], provider_options=[provider_options])
        if self.decoder_with_past is not None:
            self.decoder_with_past.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.encoder.session.get_providers()
        return self

    def can_generate(self):
        logger.warning('ORTModelForConditionalGeneration is an abstract class and is not meant to be used for generation. Please use ORTModelForSeq2SeqLM or ORTModelForSpeechSeq2Seq.')
        return False