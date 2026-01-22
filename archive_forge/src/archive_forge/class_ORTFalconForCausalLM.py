import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME
class ORTFalconForCausalLM(ORTModelForCausalLM):

    def __init__(self, model: onnxruntime.InferenceSession, config: 'PretrainedConfig', use_io_binding: Optional[bool]=None, model_save_dir: Optional[Union[str, Path, TemporaryDirectory]]=None, preprocessors: Optional[List]=None, generation_config: Optional[GenerationConfig]=None, use_cache: Optional[bool]=None, **kwargs):
        super().__init__(model=model, config=config, use_io_binding=use_io_binding, model_save_dir=model_save_dir, preprocessors=preprocessors, generation_config=generation_config, use_cache=use_cache, **kwargs)
        self.num_key_value_heads = config.num_kv_heads if config.new_decoder_architecture or not config.multi_query else 1
        self.use_alibi = config.alibi

    def _reorder_cache(self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        device_to_beam_idx = {past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past}
        reordered_past = tuple(((layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]), layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device])) for layer_past in past))
        return reordered_past

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, **kwargs) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        if not self.use_alibi and attention_mask is not None and (position_ids is None):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        return {'input_ids': input_ids, 'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask}