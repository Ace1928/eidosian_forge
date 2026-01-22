from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _binning_bucketize(confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1
    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)
    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)
    prop_bin = count_bin / count_bin.sum()
    return (acc_bin, conf_bin, prop_bin)