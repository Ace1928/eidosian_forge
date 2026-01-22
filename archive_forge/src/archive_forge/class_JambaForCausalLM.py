import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import (
from .configuration_jamba import JambaConfig
class JambaForCausalLM(JambaPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.model = JambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(JAMBA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[HybridMambaAttentionDynamicCache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_router_logits: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, num_logits_to_keep: Optional[Union[int, None]]=None) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int` or `None`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `None`, calculate logits for all
                `input_ids`. Only last token logits are needed for generation, and calculating them only for that token
                can save memory, which becomes pretty significant for long sequences.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, JambaForCausalLM

        >>> model = JambaForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_router_logits=output_router_logits, cache_position=cache_position, return_dict=return_dict)
        hidden_states = outputs[0]
        if num_logits_to_keep is None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(outputs.router_logits if return_dict else outputs[-1], self.num_experts, self.num_experts_per_tok, attention_mask)
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, output_router_logits=False, cache_position=None, **kwargs):
        empty_past_kv = past_key_values is None
        if not empty_past_kv:
            past_length = cache_position[0] if cache_position is not None else attention_mask.shape[1]
            max_cache_length = self.config.sliding_window
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if max_cache_length is not None and attention_mask is not None and (past_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(self.config, input_ids.shape[0], self.dtype, device=self.device)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask, 'output_router_logits': output_router_logits, 'num_logits_to_keep': self.config.num_logits_to_keep, 'cache_position': cache_position})
        return model_inputs