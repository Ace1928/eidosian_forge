import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
def _get_input_shape(self, input_ids=None, inputs_embeds=None):
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
    elif input_ids is not None:
        return input_ids.size()
    elif inputs_embeds is not None:
        return inputs_embeds.size()[:-1]
    else:
        raise ValueError('You have to specify either input_ids or inputs_embeds')