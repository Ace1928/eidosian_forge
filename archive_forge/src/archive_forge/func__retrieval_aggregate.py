from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat
def _retrieval_aggregate(values: Tensor, aggregation: Union[Literal['mean', 'median', 'min', 'max'], Callable]='mean', dim: Optional[int]=None) -> Tensor:
    """Aggregate the final retrieval values into a single value."""
    if aggregation == 'mean':
        return values.mean() if dim is None else values.mean(dim=dim)
    if aggregation == 'median':
        return values.median() if dim is None else values.median(dim=dim).values
    if aggregation == 'min':
        return values.min() if dim is None else values.min(dim=dim).values
    if aggregation == 'max':
        return values.max() if dim is None else values.max(dim=dim).values
    return aggregation(values, dim=dim)