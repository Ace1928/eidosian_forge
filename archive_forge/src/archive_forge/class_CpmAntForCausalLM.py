import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
@add_start_docstrings('\n    The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).\n    ', CPMANT_START_DOCSTRING)
class CpmAntForCausalLM(CpmAntPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: CpmAntConfig):
        super().__init__(config)
        self.cpmant = CpmAntModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + config.prompt_types * config.prompt_length, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                CPMAnt will process attention mask automatically, this parameter is a dummy parameter for
                text-generation pipeline.

        Example:

        Text Generation with CpmAntForCausalLM.
        ```python
        >>> from transformers import CPMAntTokenizer, CpmAntForCausalLM

        >>> texts = "今天天气不错，"
        >>> model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")
        >>> tokenizer = CPMAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
        >>> input_ids = tokenizer(texts, return_tensors="pt")
        >>> outputs = model.generate(**input_ids)
        >>> output_texts = tokenizer.batch_decode(outputs)
        >>> print(output_texts)
        ['今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的']
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_output = self.cpmant(input_ids, output_attentions, output_hidden_states, past_key_values, use_cache, return_dict)
        hidden_states = model_output.last_hidden_state if return_dict else model_output[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss_func = CrossEntropyLoss()
            loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (logits,) + model_output[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=model_output.past_key_values, hidden_states=model_output.hidden_states, attentions=model_output.attentions)

    def get_input_embeddings(self):
        return self.cpmant.input_embedding

    def set_input_embeddings(self, embeddings):
        self.cpmant.input_embedding = embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_ids = input_ids.int()
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = torch.zeros(1, 1)
        return {'input_ids': input_ids, 'use_cache': kwargs['use_cache'], 'past_key_values': kwargs.get('past_key_values', None)}

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            key_value_layer[0] = key_value_layer[0][beam_idx]
            key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values