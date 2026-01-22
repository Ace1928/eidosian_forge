import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
def _init_feed_forward_seed(self):
    """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
    if hasattr(torch.cuda, 'default_generators') and len(torch.cuda.default_generators) > 0:
        device_idx = torch.cuda.current_device()
        self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
    else:
        self.feed_forward_seed = int(torch.seed() % sys.maxsize)
    torch.manual_seed(self.feed_forward_seed)