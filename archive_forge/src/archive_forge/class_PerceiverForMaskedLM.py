import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
@add_start_docstrings('Example use of Perceiver for masked language modeling.', PERCEIVER_START_DOCSTRING)
class PerceiverForMaskedLM(PerceiverPreTrainedModel):

    def __init__(self, config: PerceiverConfig):
        super().__init__(config)
        text_preprocessor = PerceiverTextPreprocessor(config)
        trainable_position_encoding_kwargs_decoder = {'num_channels': text_preprocessor.num_channels, 'index_dims': config.max_position_embeddings}
        self.perceiver = PerceiverModel(config, input_preprocessor=text_preprocessor, decoder=PerceiverBasicDecoder(config, output_num_channels=config.d_latents, output_index_dims=config.max_position_embeddings, num_channels=text_preprocessor.num_channels, qk_channels=8 * 32, v_channels=text_preprocessor.num_channels, num_heads=8, use_query_residual=False, final_project=False, trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder))
        self.embedding_decoder = PerceiverEmbeddingDecoder(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=PerceiverMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, inputs: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None, input_ids: Optional[torch.Tensor]=None) -> Union[Tuple, PerceiverMaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, PerceiverForMaskedLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("deepmind/language-perceiver")
        >>> model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")

        >>> # training
        >>> text = "This is an incomplete sentence where some words are missing."
        >>> inputs = tokenizer(text, padding="max_length", return_tensors="pt")
        >>> # mask " missing."
        >>> inputs["input_ids"][0, 52:61] = tokenizer.mask_token_id
        >>> labels = tokenizer(text, padding="max_length", return_tensors="pt").input_ids

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> round(loss.item(), 2)
        19.87

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2048, 262]

        >>> # inference
        >>> text = "This is an incomplete sentence where some words are missing."
        >>> encoding = tokenizer(text, padding="max_length", return_tensors="pt")

        >>> # mask bytes corresponding to " missing.". Note that the model performs much better if the masked span starts with a space.
        >>> encoding["input_ids"][0, 52:61] = tokenizer.mask_token_id

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2048, 262]

        >>> masked_tokens_predictions = logits[0, 52:61].argmax(dim=-1).tolist()
        >>> tokenizer.decode(masked_tokens_predictions)
        ' missing.'
        ```"""
        if inputs is not None and input_ids is not None:
            raise ValueError('You cannot use both `inputs` and `input_ids`')
        elif inputs is None and input_ids is not None:
            inputs = input_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.perceiver(inputs=inputs, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.embedding_decoder(outputs.logits if return_dict else outputs[0], embedding_layer=self.perceiver.input_preprocessor.embeddings)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return PerceiverMaskedLMOutput(loss=masked_lm_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)