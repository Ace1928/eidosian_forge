import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_squeezebert import SqueezeBertConfig
def transpose_output(self, x):
    """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
    x = x.permute(0, 1, 3, 2).contiguous()
    new_x_shape = (x.size()[0], self.all_head_size, x.size()[3])
    x = x.view(*new_x_shape)
    return x