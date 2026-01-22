from typing import Optional, Tuple
import torch
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.file_utils import add_start_docstrings_to_model_forward
class _DecoderWithLMhead(PreTrainedModel):
    """
    Decoder model with a language modeling head on top.
    Arguments:
        model (`transformers.PreTrainedModel`):
            The model from which to extract the decoder and the language modeling head.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model.config)
        self.config = model.config
        self.decoder = model.get_decoder()
        self.lm_head = model.get_output_embeddings()
        self.final_logits_bias = getattr(model, 'final_logits_bias', None)

    @add_start_docstrings_to_model_forward(DECODER_WITH_LM_HEAD_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor, encoder_hidden_states: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None):
        decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, return_dict=True, use_cache=True)
        last_hidden_state = decoder_outputs.last_hidden_state
        if self.config.model_type == 't5' and self.config.tie_word_embeddings:
            last_hidden_state = last_hidden_state * self.config.d_model ** (-0.5)
        lm_logits = self.lm_head(last_hidden_state)
        if self.final_logits_bias is not None:
            lm_logits += self.final_logits_bias
        if labels is None:
            return (lm_logits, decoder_outputs.past_key_values)
        else:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return (loss, lm_logits, decoder_outputs.past_key_values)