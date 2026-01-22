import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longformer import LongformerConfig
@add_start_docstrings('\n    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /\n    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', LONGFORMER_START_DOCSTRING)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerQuestionAnsweringModelOutput]:
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

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForQuestionAnswering
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> encoding = tokenizer(question, text, return_tensors="pt")
        >>> input_ids = encoding["input_ids"]

        >>> # default is local attention everywhere
        >>> # the forward method will automatically set global attention on question tokens
        >>> attention_mask = encoding["attention_mask"]

        >>> outputs = model(input_ids, attention_mask=attention_mask)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        >>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
        >>> answer = tokenizer.decode(
        ...     tokenizer.convert_tokens_to_ids(answer_tokens)
        ... )  # remove space prepending space token
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            if input_ids is None:
                logger.warning('It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set.')
            else:
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
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
        return LongformerQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)