import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
@add_start_docstrings('Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4Tv2Encoder`] without embeddings and the decoder is a [`SeamlessM4Tv2TextToUnitDecoder`].', SEAMLESS_M4T_V2_START_DOCSTRING, '\n        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.\n    ')
class SeamlessM4Tv2TextToUnitModel(SeamlessM4Tv2PreTrainedModel):

    def __init__(self, config: SeamlessM4Tv2Config, embed_tokens_decoder: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4Tv2TextToUnitDecoder(config, embed_tokens_decoder)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, char_input_ids: torch.LongTensor=None, char_count_per_id: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(char_input_ids=char_input_ids, char_count_per_id=char_count_per_id, encoder_hidden_states=encoder_outputs[0], output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return SeamlessM4Tv2TextToUnitOutput(last_hidden_state=decoder_outputs.last_hidden_state, padding_mask=decoder_outputs.padding_mask, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)