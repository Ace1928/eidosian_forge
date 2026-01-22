import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@add_start_docstrings('\n    BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForQuestionAnswering(BigBirdPreTrainedModel):

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id
        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BigBirdForQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, question_lengths: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BigBirdForQuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
        """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForQuestionAnswering
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        >>> model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT

        >>> # select random article and question
        >>> LONG_ARTICLE = squad_ds[81514]["context"]
        >>> QUESTION = squad_ds[81514]["question"]
        >>> QUESTION
        'During daytime how high can the temperatures reach?'

        >>> inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")
        >>> # long article and question input
        >>> list(inputs["input_ids"].shape)
        [1, 929]

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()
        >>> predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        >>> predict_answer_token = tokenizer.decode(predict_answer_token_ids)
        ```

        ```python
        >>> target_start_index, target_end_index = torch.tensor([130]), torch.tensor([132])
        >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
        >>> loss = outputs.loss
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seqlen = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        if question_lengths is None and input_ids is not None:
            question_lengths = torch.argmax(input_ids.eq(self.sep_token_id).int(), dim=-1) + 1
            question_lengths.unsqueeze_(1)
        logits_mask = None
        if question_lengths is not None:
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = torch.ones(logits_mask.size(), dtype=int, device=logits_mask.device) - logits_mask
            logits_mask = logits_mask
            logits_mask[:, 0] = False
            logits_mask.unsqueeze_(2)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_classifier(sequence_output)
        if logits_mask is not None:
            logits = logits - logits_mask * 1000000.0
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return BigBirdForQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)
        mask = torch.where(mask < q_lengths, 1, 0)
        return mask