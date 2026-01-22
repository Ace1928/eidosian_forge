from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _drop_empty_rows_and_cols(confmat: Tensor) -> Tensor:
    """Drop all rows and columns containing only zeros.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.nominal.utils import _drop_empty_rows_and_cols
        >>> _ = torch.manual_seed(22)
        >>> matrix = torch.randint(10, size=(3, 3))
        >>> matrix[1, :] = matrix[:, 1] = 0
        >>> matrix
        tensor([[9, 0, 6],
                [0, 0, 0],
                [2, 0, 8]])
        >>> _drop_empty_rows_and_cols(matrix)
        tensor([[9, 6],
                [2, 8]])

    """
    confmat = confmat[confmat.sum(1) != 0]
    return confmat[:, confmat.sum(0) != 0]