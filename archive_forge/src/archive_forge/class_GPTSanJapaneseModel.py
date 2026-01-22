import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
@add_start_docstrings('The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.', GPTSAN_JAPANESE_START_DOCSTRING)
class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.config = copy.deepcopy(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        self.act = ACT2FN['swish']
        self.blocks = torch.nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSanJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSanJapaneseBlock(config, ext_layer=True))
        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, output_router_logits: Optional[bool]=None, num_precontext: Optional[torch.LongTensor]=None) -> Union[MoEModelOutputWithPastAndCrossAttentions, Tuple[torch.FloatTensor]]:
        """
        num_precontext (`torch.LongTensor` of shape `(batch_size,1)`):
            length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like
            BERT, tokens after that refer only to front like GPT. see also:
            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md

        Returns:
            `MoEModelOutputWithPastAndCrossAttentions` or `tuple` if `return_dict` returns
            MoEModelOutputWithPastAndCrossAttentions insted of tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = self.position_embeddings.weight.device
        if input_ids is None:
            input_ids = torch.zeros([1, 1]).int().to(device)
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        pasts_or_spout_value = None
        if past_key_values is not None:
            num_pasts_contexts = past_key_values[0][0].shape[2]
        elif self.config.d_spout and spout is not None:
            num_pasts_contexts += 1
        if self.config.d_spout and spout is not None and (attention_mask is not None):
            attention_mask_with_spout = torch.ones(num_batch, attention_mask.shape[1] + 1, device=device)
            attention_mask_with_spout[:, 1:] -= 1 - attention_mask
            attention_mask = attention_mask_with_spout
        if num_precontext is not None:
            if not (len(num_precontext.shape) == 2 and num_precontext.shape[1] == 1):
                raise ValueError('num_precontext should be [batch, 1] size.')
            num_precontext = torch.reshape(num_precontext, [-1])
        else:
            num_precontext = torch.zeros([num_batch]).int().to(device)
        num_input_contexts = input_ids.shape[1]
        num_output_contexts = num_input_contexts + num_pasts_contexts
        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is not None:
            pasts_or_spout_value = past_key_values
        elif self.config.d_spout and spout is not None:
            pasts_or_spout_value = self.spout(spout)
            pasts_or_spout_value = torch.reshape(pasts_or_spout_value, [num_batch, self.config.num_layers, 2, self.config.num_heads, num_pasts_contexts, self.config.d_model // self.config.num_heads])
            pasts_or_spout_value = torch.split(pasts_or_spout_value, [1] * self.config.num_layers, dim=1)
            pasts_or_spout_value = tuple((tuple([b.squeeze(1) for b in torch.split(a.squeeze(1), [1, 1], dim=1)]) for a in pasts_or_spout_value))
        else:
            pasts_or_spout_value = [None] * self.config.num_layers
        token_position = torch.arange(num_input_contexts).to(device) + num_pasts_contexts
        if attention_mask is None:
            attention_mask = torch.ones(num_batch, num_input_contexts, device=device)
        gather_position = (torch.zeros((num_batch, self.config.d_model, num_input_contexts)).to(device) + token_position.unsqueeze(0)).transpose(1, 2).long()
        gather_position -= (1 - attention_mask).argmin(dim=-1).unsqueeze(1).unsqueeze(2)
        gather_position = torch.clip(gather_position, num_pasts_contexts, self.config.max_position_embeddings - 1)
        for i in range(num_batch):
            hidden_states[i] += torch.gather(self.position_embeddings.weight, dim=0, index=gather_position[i])
        causal_mask = torch.tril(torch.ones((num_output_contexts, num_output_contexts), dtype=torch.uint8)).view(1, 1, num_output_contexts, num_output_contexts).to(device)
        prefix_lm_mask = causal_mask[:, :, -num_input_contexts:, :]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).unsqueeze(2)
            prefix_lm_mask = (prefix_lm_mask + token_type_ids > 0).float()
        extended_attention_mask = prefix_lm_mask * attention_mask.unsqueeze(1).unsqueeze(2)
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_switch_layers + self.config.num_ext_layers)
        present_key_value_states = () if self.config.use_cache or use_cache else None
        all_hidden_states = () if self.config.output_hidden_states or output_hidden_states else None
        all_attentions = () if self.config.output_attentions or output_attentions else None
        all_router_probs = () if self.config.output_router_logits or output_router_logits else None
        for layer, past in enumerate(pasts_or_spout_value):
            if layer == self.config.num_switch_layers:
                if self.config.num_ext_layers > 0:
                    for i in range(num_batch):
                        hidden_states[i] += torch.gather(self.extra_position_embeddings.weight, dim=0, index=gather_position[i])
            output_router_tuple = (self.config.output_router_logits or output_router_logits) and layer < self.config.num_switch_layers
            block_output = self.blocks[layer](hidden_states=hidden_states, past_key_value=past, attention_mask=extended_attention_mask, head_mask=head_mask, use_cache=self.config.use_cache or use_cache, output_attentions=self.config.output_attentions or output_attentions, output_router_tuple=output_router_tuple)
            outpos = 0
            hidden_states = block_output[outpos]
            if self.config.output_hidden_states or output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.use_cache or use_cache:
                outpos += 1
                present = block_output[outpos]
                present_key_value_states += (present,)
            if self.config.output_attentions or output_attentions:
                outpos += 1
                attention_probs = block_output[outpos]
                all_attentions += (attention_probs,)
            if output_router_tuple:
                outpos += 1
                router_tuple = block_output[outpos]
                all_router_probs.append(router_tuple[0])
        hidden_states = self.last_project(hidden_states)
        hidden_states = self.act(hidden_states)
        if self.config.output_hidden_states or output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_router_probs] if v is not None))
        return MoEModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, router_probs=all_router_probs)