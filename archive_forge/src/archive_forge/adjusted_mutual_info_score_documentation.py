from typing import Literal
import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.mutual_info_score import _mutual_info_score_compute, _mutual_info_score_update
from torchmetrics.functional.clustering.utils import (
Calculated expected mutual information score between two clusterings.

    Implementation taken from sklearn/metrics/cluster/_expected_mutual_info_fast.pyx.

    Args:
        contingency: contingency matrix
        n_samples: number of samples

    Returns:
        expected_mutual_info_score: expected mutual information score

    