import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
@add_start_docstrings('The NllbMoe Model with a language modeling head. Can be used for summarization.', NLLB_MOE_START_DOCSTRING)
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel):
    base_model_prefix = 'model'
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)
        self.model = NllbMoeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(NLLB_MOE_GENERATION_EXAMPLE)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_router_logits: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], Seq2SeqMoEOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_router_logits=output_router_logits, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0])
        loss = None
        encoder_aux_loss = None
        decoder_aux_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            if output_router_logits:
                encoder_router_logits = outputs[-1]
                decoder_router_logits = outputs[3 if output_attentions else 4]
                encoder_router_logits, encoder_expert_indexes = self._unpack_router_logits(encoder_router_logits)
                encoder_aux_loss = load_balancing_loss_func(encoder_router_logits, encoder_expert_indexes)
                decoder_router_logits, decoder_expert_indexes = self._unpack_router_logits(decoder_router_logits)
                decoder_aux_loss = load_balancing_loss_func(decoder_router_logits, decoder_expert_indexes)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if output_router_logits and labels is not None:
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + aux_loss
        output = (loss,) if loss is not None else ()
        if not return_dict:
            output += (lm_logits,)
            if output_router_logits:
                output += (encoder_aux_loss, decoder_aux_loss, *outputs[1:])
            else:
                output += outputs[1:]
            return output
        return Seq2SeqMoEOutput(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, cross_attentions=outputs.cross_attentions, encoder_aux_loss=encoder_aux_loss, decoder_aux_loss=decoder_aux_loss, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, decoder_hidden_states=outputs.decoder_hidden_states, encoder_attentions=outputs.encoder_attentions, decoder_attentions=outputs.decoder_attentions, encoder_router_logits=outputs.encoder_router_logits, decoder_router_logits=outputs.decoder_router_logits)

    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if router_output is not None:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        total_router_logits = torch.cat(total_router_logits, dim=1) if len(total_router_logits) > 0 else None
        total_expert_indexes = torch.stack(total_expert_indexes, dim=1) if len(total_expert_indexes) > 0 else None
        return (total_router_logits, total_expert_indexes)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past