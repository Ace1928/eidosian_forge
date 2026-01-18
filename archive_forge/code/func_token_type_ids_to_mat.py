import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor:
    """Convert `token_type_ids` to `token_type_mat`."""
    token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
    cls_ids = token_type_ids == self.cls_token_type_id
    cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
    return cls_mat | token_type_mat