import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
class GraphormerDecoderHead(nn.Module):

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        'num_classes should be 1 for regression, or the number of classes for classification'
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes