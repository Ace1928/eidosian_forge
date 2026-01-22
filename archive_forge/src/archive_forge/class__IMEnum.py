import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
@unique
class _IMEnum(EnumStr):
    """A helper Enum class for storing the information measure."""

    @staticmethod
    def _name() -> str:
        return 'Information measure'
    KL_DIVERGENCE = 'kl_divergence'
    ALPHA_DIVERGENCE = 'alpha_divergence'
    BETA_DIVERGENCE = 'beta_divergence'
    AB_DIVERGENCE = 'ab_divergence'
    RENYI_DIVERGENCE = 'renyi_divergence'
    L1_DISTANCE = 'l1_distance'
    L2_DISTANCE = 'l2_distance'
    L_INFINITY_DISTANCE = 'l_infinity_distance'
    FISHER_RAO_DISTANCE = 'fisher_rao_distance'