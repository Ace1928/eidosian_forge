import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_albert import AlbertConfig
@add_start_docstrings('\n    Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a\n    `sentence order prediction (classification)` head.\n    ', ALBERT_START_DOCSTRING)
class AlbertForPreTraining(AlbertPreTrainedModel):
    _tied_weights_keys = ['predictions.decoder.bias', 'predictions.decoder.weight']

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=AlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, sentence_order_label: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[AlbertForPreTrainingOutput, Tuple]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        sentence_order_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`. `0` indicates original order (sequence A, then
            sequence B), `1` indicates switched order (sequence B, then sequence A).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AlbertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = AlbertForPreTraining.from_pretrained("albert/albert-base-v2")

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)
        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss
        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return AlbertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, sop_logits=sop_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)