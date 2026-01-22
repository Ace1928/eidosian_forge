from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _del_column(data: Tensor, idx: int) -> Tensor:
    """Delete the column at index."""
    return torch.cat([data[:, :idx], data[:, idx + 1:]], 1)