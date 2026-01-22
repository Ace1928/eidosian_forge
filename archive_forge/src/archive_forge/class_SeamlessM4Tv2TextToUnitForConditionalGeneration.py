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
@add_start_docstrings('Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4Tv2TextToUnitModel`].', SEAMLESS_M4T_V2_START_DOCSTRING, '\n        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.\n    ')
class SeamlessM4Tv2TextToUnitForConditionalGeneration(SeamlessM4Tv2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ['vocoder', 'speech_encoder', 'text_encoder', 'text_decoder']
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: SeamlessM4Tv2Config, embed_tokens_decoder: Optional[nn.Embedding]=None):
        config = copy.deepcopy(config)
        for param, val in config.to_dict().items():
            if param.startswith('t2u_'):
                config.__setattr__(param[4:], val)
        super().__init__(config)
        self.model = SeamlessM4Tv2TextToUnitModel(config, embed_tokens_decoder)
        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    @add_start_docstrings_to_model_forward(M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, char_input_ids: torch.LongTensor=None, char_count_per_id: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids, char_input_ids=char_input_ids, char_count_per_id=char_count_per_id, attention_mask=attention_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return SeamlessM4Tv2TextToUnitOutput(last_hidden_state=lm_logits, padding_mask=outputs.padding_mask, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions, loss=masked_lm_loss)

    def _tie_weights(self) -> None:
        if getattr(self.config, 'tie_word_embeddings', True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())