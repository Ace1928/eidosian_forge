from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
Compute explained variance.

    Args:
        preds: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings):

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Example:
        >>> from torchmetrics.functional.regression import explained_variance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance(preds, target, multioutput='raw_values')
        tensor([0.9677, 1.0000])

    