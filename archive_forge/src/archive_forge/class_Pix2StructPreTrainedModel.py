import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
class Pix2StructPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Pix2StructConfig

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'decoder_input_ids': input_ids, 'input_ids': input_ids, 'decoder_attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, Pix2StructLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, Pix2StructTextDenseGatedActDense):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            d_ff = self.config.text_config.d_ff if isinstance(self.config, Pix2StructConfig) else self.config.d_ff
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pix2StructTextAttention):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            key_value_proj_dim = self.config.text_config.d_kv if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            n_heads = self.config.text_config.num_heads if isinstance(self.config, Pix2StructConfig) else self.config.num_heads
            module.query.weight.data.normal_(mean=0.0, std=factor * (hidden_size * key_value_proj_dim) ** (-0.5))
            module.key.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            module.value.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            module.output.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
        elif isinstance(module, nn.Embedding):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            module.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Pix2StructTextModel):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            module.lm_head.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Pix2StructLayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id. See Pix2Struct docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids