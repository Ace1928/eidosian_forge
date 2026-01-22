from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
def _generate_dummy_inputs_for_default_and_seq2seq_lm(self, tokenizer: PreTrainedTokenizer, batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
    encoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(tokenizer, batch_size, seq_length, is_pair, framework)
    decoder_seq_length = seq_length if not self.use_past else 1
    decoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(tokenizer, batch_size, decoder_seq_length, is_pair, framework)
    decoder_inputs = {f'decoder_{name}': tensor for name, tensor in decoder_inputs.items()}
    common_inputs = dict(**encoder_inputs, **decoder_inputs)
    if self.use_past:
        if not is_torch_available():
            raise ValueError('Cannot generate dummy past_keys inputs without PyTorch installed.')
        else:
            import torch
        batch, encoder_seq_length = common_inputs['input_ids'].shape
        decoder_seq_length = common_inputs['decoder_input_ids'].shape[1]
        num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
        encoder_shape = (batch, num_encoder_attention_heads, encoder_seq_length, self._config.hidden_size // num_encoder_attention_heads)
        decoder_past_length = decoder_seq_length + 3
        decoder_shape = (batch, num_decoder_attention_heads, decoder_past_length, self._config.hidden_size // num_decoder_attention_heads)
        common_inputs['decoder_attention_mask'] = torch.cat([common_inputs['decoder_attention_mask'], torch.ones(batch, decoder_past_length)], dim=1)
        common_inputs['past_key_values'] = []
        num_encoder_layers, num_decoder_layers = self.num_layers
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        remaining_side_name = 'encoder' if num_encoder_layers > num_decoder_layers else 'decoder'
        for _ in range(min_num_layers):
            common_inputs['past_key_values'].append((torch.zeros(decoder_shape), torch.zeros(decoder_shape), torch.zeros(encoder_shape), torch.zeros(encoder_shape)))
        shape = encoder_shape if remaining_side_name == 'encoder' else decoder_shape
        for _ in range(min_num_layers, max_num_layers):
            common_inputs['past_key_values'].append((torch.zeros(shape), torch.zeros(shape)))
    return common_inputs