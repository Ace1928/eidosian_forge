import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_electra import ElectraConfig
@add_start_docstrings('\n    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.\n\n    It is recommended to load the discriminator checkpoint into that model.\n    ', ELECTRA_START_DOCSTRING)
class ElectraForPreTraining(ElectraPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=ElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], ElectraForPreTrainingOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see `input_ids` docstring)
            Indices should be in `[0, 1]`:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples:

        ```python
        >>> from transformers import ElectraForPreTraining, AutoTokenizer
        >>> import torch

        >>> discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")

        >>> sentence = "The quick brown fox jumps over the lazy dog"
        >>> fake_sentence = "The quick brown fox fake over the lazy dog"

        >>> fake_tokens = tokenizer.tokenize(fake_sentence, add_special_tokens=True)
        >>> fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
        >>> discriminator_outputs = discriminator(fake_inputs)
        >>> predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

        >>> fake_tokens
        ['[CLS]', 'the', 'quick', 'brown', 'fox', 'fake', 'over', 'the', 'lazy', 'dog', '[SEP]']

        >>> predictions.squeeze().tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        discriminator_hidden_states = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return ElectraForPreTrainingOutput(loss=loss, logits=logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)