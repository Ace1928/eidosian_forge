from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
def _generate_dummy_inputs_for_causal_lm(self, tokenizer: PreTrainedTokenizer, batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
    common_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(tokenizer, batch_size, seq_length, is_pair, framework)
    if self.use_past:
        if not is_torch_available():
            raise ValueError('Cannot generate dummy past_keys inputs without PyTorch installed.')
        else:
            import torch
        batch, seqlen = common_inputs['input_ids'].shape
        past_key_values_length = seqlen + 2
        num_encoder_layers, _ = self.num_layers
        num_encoder_attention_heads, _ = self.num_attention_heads
        past_shape = (batch, num_encoder_attention_heads, past_key_values_length, self._config.hidden_size // num_encoder_attention_heads)
        mask_dtype = common_inputs['attention_mask'].dtype
        common_inputs['attention_mask'] = torch.cat([common_inputs['attention_mask'], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1)
        common_inputs['past_key_values'] = [(torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)]
    return common_inputs