import itertools
from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
def theils_u_matrix(matrix: Tensor, nan_strategy: Literal['replace', 'drop']='replace', nan_replace_value: Optional[float]=0.0) -> Tensor:
    """Compute `Theil's U`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Theil's U statistic for analyses of correlation between categorical
    variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:
            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Theil's U statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import theils_u_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> theils_u_matrix(matrix)
        tensor([[1.0000, 0.0202, 0.0142, 0.0196, 0.0353],
                [0.0202, 1.0000, 0.0070, 0.0136, 0.0065],
                [0.0143, 0.0070, 1.0000, 0.0125, 0.0206],
                [0.0198, 0.0137, 0.0125, 1.0000, 0.0312],
                [0.0352, 0.0065, 0.0204, 0.0308, 1.0000]])

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    theils_u_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = (matrix[:, i], matrix[:, j])
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _theils_u_update(x, y, num_classes, nan_strategy, nan_replace_value)
        theils_u_matrix_value[i, j] = _theils_u_compute(confmat)
        theils_u_matrix_value[j, i] = _theils_u_compute(confmat.T)
    return theils_u_matrix_value