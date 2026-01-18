from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
Compute the label ranking loss for multilabel data [1].

    The score is corresponds to the average number of label pairs that are incorrectly ordered given some predictions
    weighted by the size of the label set and the number of labels not in the label set. The best score is 0.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.functional.classification import multilabel_ranking_loss
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> multilabel_ranking_loss(preds, target, num_labels=5)
        tensor(0.4167)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.

    