import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig

        start_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        question_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            The positions of all question tokens. If given, start_logits and end_logits will be of shape `(batch_size,
            num_questions, sequence_length)`. If None, the first question token in each sequence in the batch will be
            the only one for which start_logits and end_logits are calculated and they will be of shape `(batch_size,
            sequence_length)`.
        