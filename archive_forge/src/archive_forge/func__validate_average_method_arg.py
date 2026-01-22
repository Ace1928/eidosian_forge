from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _validate_average_method_arg(average_method: Literal['min', 'geometric', 'arithmetic', 'max']='arithmetic') -> None:
    if average_method not in ('min', 'geometric', 'arithmetic', 'max'):
        raise ValueError(f'Expected argument `average_method` to be one of  `min`, `geometric`, `arithmetic`, `max`,but got {average_method}')