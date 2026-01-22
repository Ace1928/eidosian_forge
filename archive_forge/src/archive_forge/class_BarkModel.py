import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@add_start_docstrings("\n    The full Bark model, a text-to-speech model composed of 4 sub-models:\n    - [`BarkSemanticModel`] (also referred to as the 'text' model): a causal auto-regressive transformer model that\n      takes\n    as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.\n    - [`BarkCoarseModel`] (also refered to as the 'coarse acoustics' model), also a causal autoregressive transformer,\n    that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary\n    to `encodec`.\n    - [`BarkFineModel`] (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively\n    predicts the last codebooks based on the sum of the previous codebooks embeddings.\n    - having predicted all the codebook channels from the [`EncodecModel`], Bark uses it to decode the output audio\n      array.\n\n    It should be noted that each of the first three modules can support conditional speaker embeddings to condition the\n    output sound according to specific predefined voice.\n    ", BARK_START_DOCSTRING)
class BarkModel(BarkPreTrainedModel):
    config_class = BarkConfig

    def __init__(self, config):
        super().__init__(config)
        self.semantic = BarkSemanticModel(config.semantic_config)
        self.coarse_acoustics = BarkCoarseModel(config.coarse_acoustics_config)
        self.fine_acoustics = BarkFineModel(config.fine_acoustics_config)
        self.codec_model = AutoModel.from_config(config.codec_config)
        self.config = config

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        if not hasattr(self.semantic, '_hf_hook'):
            return get_parameter_device(self)
        for module in self.semantic.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)

    def enable_cpu_offload(self, gpu_id: Optional[int]=0):
        """
        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until
        the next sub-model runs.

        Args:
            gpu_id (`int`, *optional*, defaults to 0):
                GPU id on which the sub-models will be loaded and offloaded.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError('`enable_model_cpu_offload` requires `accelerate`.')
        device = torch.device(f'cuda:{gpu_id}')
        if self.device.type != 'cpu':
            self.to('cpu')
            torch.cuda.empty_cache()
        self.semantic.input_embeds_layer, _ = cpu_offload_with_hook(self.semantic.input_embeds_layer, device)
        hook = None
        for cpu_offloaded_model in [self.semantic, self.coarse_acoustics, self.fine_acoustics]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
        self.fine_acoustics_hook = hook
        _, hook = cpu_offload_with_hook(self.codec_model, device, prev_module_hook=hook)
        self.codec_model_hook = hook

    def codec_decode(self, fine_output, output_lengths=None):
        """Turn quantized audio codes into audio array using encodec."""
        fine_output = fine_output.transpose(0, 1)
        emb = self.codec_model.quantizer.decode(fine_output)
        if output_lengths is not None:
            out = [sample[:, :l].unsqueeze(0) for sample, l in zip(emb, output_lengths)]
            audio_arr = [self.codec_model.decoder(sample).squeeze() for sample in out]
        else:
            out = self.codec_model.decoder(emb)
            audio_arr = out.squeeze(1)
        return audio_arr

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor]=None, history_prompt: Optional[Dict[str, torch.Tensor]]=None, return_output_lengths: Optional[bool]=None, **kwargs) -> torch.LongTensor:
        """
        Generates audio from an input prompt and an additional optional `Bark` speaker prompt.

        Args:
            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the
                longest generation among the batch.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.
            kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *semantic_*, *coarse_*, *fine_* prefix, they will be input for the `generate` method of the
                semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for all sub-models except one.
            return_output_lengths (`bool`, *optional*):
                Whether or not to return the waveform lengths. Useful when batching.
        Returns:
            By default:
                - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.
            When `return_output_lengths=True`:
                Returns a tuple made of:
                - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.
                - **output_lengths** (`torch.Tensor` of shape (batch_size)): The length of each waveform in the batch
        Example:

        ```python
        >>> from transformers import AutoProcessor, BarkModel

        >>> processor = AutoProcessor.from_pretrained("suno/bark-small")
        >>> model = BarkModel.from_pretrained("suno/bark-small")

        >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
        >>> voice_preset = "v2/en_speaker_6"

        >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)

        >>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
        >>> audio_array = audio_array.cpu().numpy().squeeze()
        ```
        """
        semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
        coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
        fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)
        kwargs_semantic = {'attention_mask': kwargs.pop('attention_mask', None), 'min_eos_p': kwargs.pop('min_eos_p', None)}
        kwargs_coarse = {}
        kwargs_fine = {}
        for key, value in kwargs.items():
            if key.startswith('semantic_'):
                key = key[len('semantic_'):]
                kwargs_semantic[key] = value
            elif key.startswith('coarse_'):
                key = key[len('coarse_'):]
                kwargs_coarse[key] = value
            elif key.startswith('fine_'):
                key = key[len('fine_'):]
                kwargs_fine[key] = value
            else:
                if key not in kwargs_semantic:
                    kwargs_semantic[key] = value
                if key not in kwargs_coarse:
                    kwargs_coarse[key] = value
                if key not in kwargs_fine:
                    kwargs_fine[key] = value
        semantic_output = self.semantic.generate(input_ids, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, **kwargs_semantic)
        coarse_output = self.coarse_acoustics.generate(semantic_output, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, coarse_generation_config=coarse_generation_config, codebook_size=self.generation_config.codebook_size, return_output_lengths=return_output_lengths, **kwargs_coarse)
        output_lengths = None
        if return_output_lengths:
            coarse_output, output_lengths = coarse_output
            output_lengths = output_lengths // coarse_generation_config.n_coarse_codebooks
        output = self.fine_acoustics.generate(coarse_output, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, coarse_generation_config=coarse_generation_config, fine_generation_config=fine_generation_config, codebook_size=self.generation_config.codebook_size, **kwargs_fine)
        if getattr(self, 'fine_acoustics_hook', None) is not None:
            self.fine_acoustics_hook.offload()
            self.codec_model = self.codec_model.to(self.device)
        audio = self.codec_decode(output, output_lengths)
        if getattr(self, 'codec_model_hook', None) is not None:
            self.codec_model_hook.offload()
        if return_output_lengths:
            output_lengths = [len(sample) for sample in audio]
            audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
            return (audio, output_lengths)
        return audio

    @classmethod
    def _check_and_enable_flash_attn_2(cls, config, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None, hard_check_only: bool=False):
        """
        `_check_and_enable_flash_attn_2` originally don't expand flash attention enabling to the model
        sub-configurations. We override the original method to make sure that Bark sub-models are using Flash Attention
        if necessary.

        If you don't know about Flash Attention, check out the official repository of flash attention:
        https://github.com/Dao-AILab/flash-attention

        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this
        specific section of the documentation to learn more about it:
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models

        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in
        half precision and not ran on CPU.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flash_attention_2" so that the model
        can initialize the correct attention module
        """
        config = super()._check_and_enable_flash_attn_2(config, torch_dtype, device_map, hard_check_only=hard_check_only)
        config.semantic_config._attn_implementation = config._attn_implementation
        config.coarse_acoustics_config._attn_implementation = config._attn_implementation
        config.fine_acoustics_config._attn_implementation = config._attn_implementation
        return config