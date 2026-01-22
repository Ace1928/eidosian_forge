import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = 'wav2vec2'
    main_input_name = 'input_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Wav2Vec2ForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(module.conv.weight, mean=0, std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool]=None):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_adapters(self):
        if self.config.adapter_attn_dim is None:
            raise ValueError(f'{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.')
        adapter_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights['.'.join([name, param_name])] = param
        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_weights['.'.join(['lm_head', name])] = param
        return adapter_weights

    def init_adapter_layers(self):
        """
        (Re-)initialize attention adapter layers and lm head for adapter-only fine-tuning
        """
        for module in self.modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)

    def load_adapter(self, target_lang: str, force_load=True, **kwargs):
        """
        Load a language adapter model from a pre-trained adapter model.

        Parameters:
            target_lang (`str`):
                Has to be a language id of an existing adapter weight. Adapter weights are stored in the format
                adapter.<lang>.safetensors or adapter.<lang>.bin
            force_load (`bool`, defaults to `True`):
                Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import Wav2Vec2ForCTC, AutoProcessor

        >>> ckpt = "facebook/mms-1b-all"
        >>> processor = AutoProcessor.from_pretrained(ckpt)
        >>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
        >>> # set specific language
        >>> processor.tokenizer.set_target_lang("spa")
        >>> model.load_adapter("spa")
        ```
        """
        if self.config.adapter_attn_dim is None:
            raise ValueError(f'Cannot load_adapter for {target_lang} if `config.adapter_attn_dim` is not defined.')
        if target_lang == self.target_lang and (not force_load):
            logger.warning(f'Adapter weights are already set to {target_lang}.')
            return
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        token = kwargs.pop('token', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        revision = kwargs.pop('revision', None)
        use_safetensors = kwargs.pop('use_safetensors', None if is_safetensors_available() else False)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        model_path_or_id = self.config._name_or_path
        state_dict = None
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
            try:
                weight_path = cached_file(model_path_or_id, filename=filepath, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, cache_dir=cache_dir)
                state_dict = safe_load_file(weight_path)
            except EnvironmentError:
                if use_safetensors:
                    raise
            except Exception:
                if use_safetensors:
                    raise EnvironmentError(f"Can't load the model for '{model_path_or_id}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a directory containing a file named {filepath}.")
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)
            try:
                weight_path = cached_file(model_path_or_id, filename=filepath, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, cache_dir=cache_dir)
                weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
                state_dict = torch.load(weight_path, map_location='cpu', **weights_only_kwarg)
            except EnvironmentError:
                raise
            except Exception:
                raise EnvironmentError(f"Can't load the model for '{model_path_or_id}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a directory containing a file named {filepath}.")
        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())
        if len(unexpected_keys) > 0:
            raise ValueError(f'The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.')
        elif len(missing_keys) > 0:
            raise ValueError(f'The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.')
        target_vocab_size = state_dict['lm_head.weight'].shape[0]
        if target_vocab_size != self.config.vocab_size:
            self.lm_head = nn.Linear(self.config.output_hidden_size, target_vocab_size, device=self.device, dtype=self.dtype)
            self.config.vocab_size = target_vocab_size
        state_dict = {k: v.to(adapter_weights[k]) for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        self.target_lang = target_lang