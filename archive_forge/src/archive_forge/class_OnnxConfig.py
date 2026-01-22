import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from packaging import version
from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size
class OnnxConfig(ABC):
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.
    """
    default_fixed_batch = 2
    default_fixed_sequence = 8
    default_fixed_num_choices = 4
    torch_onnx_minimum_version = version.parse('1.8')
    _tasks_to_common_outputs = {'causal-lm': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'default': OrderedDict({'last_hidden_state': {0: 'batch', 1: 'sequence'}}), 'image-classification': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'image-segmentation': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}, 'pred_boxes': {0: 'batch', 1: 'sequence'}, 'pred_masks': {0: 'batch', 1: 'sequence'}}), 'masked-im': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'masked-lm': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'multiple-choice': OrderedDict({'logits': {0: 'batch'}}), 'object-detection': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}, 'pred_boxes': {0: 'batch', 1: 'sequence'}}), 'question-answering': OrderedDict({'start_logits': {0: 'batch', 1: 'sequence'}, 'end_logits': {0: 'batch', 1: 'sequence'}}), 'semantic-segmentation': OrderedDict({'logits': {0: 'batch', 1: 'num_labels', 2: 'height', 3: 'width'}}), 'seq2seq-lm': OrderedDict({'logits': {0: 'batch', 1: 'decoder_sequence'}}), 'sequence-classification': OrderedDict({'logits': {0: 'batch'}}), 'token-classification': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'vision2seq-lm': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}}), 'speech2seq-lm': OrderedDict({'logits': {0: 'batch', 1: 'sequence'}})}

    def __init__(self, config: 'PretrainedConfig', task: str='default', patching_specs: List[PatchingSpec]=None):
        self._config = config
        if task not in self._tasks_to_common_outputs:
            raise ValueError(f'{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}')
        self.task = task
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    @classmethod
    def from_model_config(cls, config: 'PretrainedConfig', task: str='default') -> 'OnnxConfig':
        """
        Instantiate a OnnxConfig for a specific model

        Args:
            config: The model's configuration to use when exporting to ONNX

        Returns:
            OnnxConfig for this model
        """
        return cls(config, task=task)

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model

        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        if hasattr(self._config, 'use_cache'):
            return {'use_cache': False}
        return None

    @property
    def default_batch_size(self) -> int:
        """
        The default batch size to use if no other indication

        Returns:
            Integer > 0
        """
        return OnnxConfig.default_fixed_batch

    @property
    def default_sequence_length(self) -> int:
        """
        The default sequence length to use if no other indication

        Returns:
            Integer > 0
        """
        return OnnxConfig.default_fixed_sequence

    @property
    def default_num_choices(self) -> int:
        """
        The default number of choices to use if no other indication

        Returns:
            Integer > 0
        """
        return OnnxConfig.default_fixed_num_choices

    @property
    def default_onnx_opset(self) -> int:
        """
        Which onnx opset to use when exporting the model

        Returns:
            Integer ONNX Opset version
        """
        return DEFAULT_ONNX_OPSET

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-05

    @property
    def is_torch_support_available(self) -> bool:
        """
        The minimum PyTorch version required to export the model.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        if is_torch_available():
            from transformers.utils import get_torch_version
            return version.parse(get_torch_version()) >= self.torch_onnx_minimum_version
        else:
            return False

    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameter on the model

        Returns:
            True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
        """
        return compute_serialized_parameters_size(num_parameters, ParameterFormat.Float) >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT

    def _generate_dummy_images(self, batch_size: int=2, num_channels: int=3, image_height: int=40, image_width: int=40):
        images = []
        for _ in range(batch_size):
            data = np.random.rand(image_height, image_width, num_channels) * 255
            images.append(Image.fromarray(data.astype('uint8')).convert('RGB'))
        return images

    def _generate_dummy_audio(self, batch_size: int=2, sampling_rate: int=22050, time_duration: float=5.0, frequency: int=220):
        audio_data = []
        for _ in range(batch_size):
            t = np.linspace(0, time_duration, int(time_duration * sampling_rate), endpoint=False)
            audio_data.append(0.5 * np.sin(2 * np.pi * frequency * t))
        return audio_data

    def generate_dummy_inputs(self, preprocessor: Union['PreTrainedTokenizerBase', 'FeatureExtractionMixin', 'ImageProcessingMixin'], batch_size: int=-1, seq_length: int=-1, num_choices: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None, num_channels: int=3, image_width: int=40, image_height: int=40, sampling_rate: int=22050, time_duration: float=5.0, frequency: int=220, tokenizer: 'PreTrainedTokenizerBase'=None) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            preprocessor: ([`PreTrainedTokenizerBase`], [`FeatureExtractionMixin`], or [`ImageProcessingMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            num_choices (`int`, *optional*, defaults to -1):
                The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.
            sampling_rate (`int`, *optional* defaults to 22050)
                The sampling rate for audio data generation.
            time_duration (`float`, *optional* defaults to 5.0)
                Total seconds of sampling for audio data generation.
            frequency (`int`, *optional* defaults to 220)
                The desired natural frequency of generated audio.

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        from ..feature_extraction_utils import FeatureExtractionMixin
        from ..image_processing_utils import ImageProcessingMixin
        from ..tokenization_utils_base import PreTrainedTokenizerBase
        if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
            raise ValueError('You cannot provide both a tokenizer and a preprocessor to generate dummy inputs.')
        if tokenizer is not None:
            warnings.warn('The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use `preprocessor` instead.', FutureWarning)
            logger.warning('Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.')
            preprocessor = tokenizer
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0)
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add)
            input_token = preprocessor.unk_token if preprocessor.unk_token is not None and len(preprocessor.unk_token) > 0 else '0'
            dummy_input = [' '.join([input_token]) * seq_length] * batch_size
            if self.task == 'multiple-choice':
                num_choices = compute_effective_axis_dimension(num_choices, fixed_dimension=OnnxConfig.default_fixed_num_choices, num_token_to_add=0)
                dummy_input = dummy_input * num_choices
                tokenized_input = preprocessor(dummy_input, text_pair=dummy_input)
                for k, v in tokenized_input.items():
                    tokenized_input[k] = [v[i:i + num_choices] for i in range(0, len(v), num_choices)]
                return dict(tokenized_input.convert_to_tensors(tensor_type=framework))
            return dict(preprocessor(dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, ImageProcessingMixin):
            if preprocessor.model_input_names[0] != 'pixel_values':
                raise ValueError(f'The `preprocessor` is an image processor ({preprocessor.__class__.__name__}) and expects `model_input_names[0]` to be "pixel_values", but got {preprocessor.model_input_names[0]}')
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == 'pixel_values':
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == 'input_features':
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_audio(batch_size, sampling_rate, time_duration, frequency)
            return dict(preprocessor(dummy_input, return_tensors=framework))
        else:
            raise ValueError('Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor.')

    def generate_dummy_inputs_onnxruntime(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generate inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq
        models which have the encoder and decoder exported as separate ONNX files.

        Args:
            reference_model_inputs ([`Mapping[str, Tensor]`):
                Reference inputs for the model.

        Returns:
            `Mapping[str, Tensor]`: The mapping holding the kwargs to provide to the model's forward function
        """
        return reference_model_inputs

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        """
        Flatten any potential nested structure expanding the name of the field with the index of the element within the
        structure.

        Args:
            name: The name of the nested structure
            field: The structure to, potentially, be flattened

        Returns:
            (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

        """
        from itertools import chain
        return {f'{name}.{idx}': item for idx, item in enumerate(chain.from_iterable(field))}