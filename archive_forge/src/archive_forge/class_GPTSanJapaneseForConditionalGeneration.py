import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
@add_start_docstrings('The bare GPTSAN-japanese Model with a language modeling head.', GPTSAN_JAPANESE_START_DOCSTRING)
class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        self.model = GPTSanJapaneseModel(config)
        self.register_buffer('final_logits_bias', torch.zeros([1, config.vocab_size]))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, output_router_logits: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.FloatTensor], MoECausalLMOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
            `MoECausalLMOutputWithPast` or `tuple` if `return_dict` returns MoECausalLMOutputWithPast insted of tuple

        Example:

        Text Generation with regular LM Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("織田信長は、", return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> gen_token = model.generate(input_ids, max_new_tokens=50)
        >>> tokenizer.decode(gen_token[0])
        "織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."
        ```

        Text Generation with Prefix-LM Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("", prefix_text="織田信長は、", return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> token_type_ids = x_token.token_type_ids.to(device)
        >>> gen_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
        >>> tokenizer.decode(gen_token[0])
        "織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."
        ```

        Simultaneously Text Generation And Masked Language Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> masked_sentence = "武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。"
        >>> x_token = tokenizer("", prefix_text=masked_sentence, return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> token_type_ids = x_token.token_type_ids.to(device)
        >>> out_lm_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
        >>> out_mlm_token = model(input_ids, token_type_ids=token_type_ids).logits.argmax(axis=-1)
        >>> tokenizer.decode(out_mlm_token[0])
        "武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"

        >>> tokenizer.decode(out_lm_token[0][input_ids.shape[1] :])
        "武田氏の三代に渡った武田家のひとり\\n甲斐市に住む、日本史上最大の戦国大名。..."
        ```"""
        SEG_TOKEN = self.config.separator_token_id
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_return_dict = True
        num_precontext = None
        if input_ids is not None:
            num_batch = input_ids.shape[0]
            num_precontext = torch.zeros([num_batch]).int().to(input_ids.device)
            where_separators = torch.where(input_ids == SEG_TOKEN)
            num_precontext[where_separators[0]] += where_separators[1]
            num_precontext = num_precontext.unsqueeze(1)
        outputs = self.model(input_ids, attention_mask, token_type_ids, spout, past_key_values, head_mask, use_cache, inputs_embeds, decoder_inputs_embeds, output_attentions, output_hidden_states, model_return_dict, output_router_logits, num_precontext)
        lm_logits = self.lm_head(outputs[0])
        if lm_logits.shape[-1] == self.final_logits_bias.shape[-1]:
            lm_logits = lm_logits + self.final_logits_bias
        loss = None
        z_loss = None
        router_probs = None
        aux_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if output_router_logits:
                router_logits, expert_indexes = self._unpack_router_logits(outputs.router_probs)
                z_loss = router_z_loss_func(router_logits)
                router_probs = nn.Softmax(dim=-1)(router_logits)
                aux_loss = load_balancing_loss_func(router_probs, expert_indexes)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            return tuple((v for v in [loss, lm_logits, outputs.past_key_values, outputs.hidden_states, outputs.router_probs, z_loss, aux_loss] if v is not None))
        return MoECausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_probs, z_loss=z_loss, aux_loss=aux_loss)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[Union[List, torch.FloatTensor]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, **kwargs):
        if isinstance(spout, list):
            spout = torch.tensor(spout).float()
            if input_ids is not None:
                spout = spout.to(input_ids.device)
        if past_key_values is not None:
            return {'input_ids': input_ids[:, -1:] if input_ids is not None else None, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids[:, -1:] if token_type_ids is not None else None, 'spout': spout, 'past_key_values': past_key_values}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'spout': spout, 'past_key_values': None}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer('final_logits_bias', new_bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        return (torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1))