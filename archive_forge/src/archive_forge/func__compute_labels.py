import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _compute_labels(self, F, batch_size):
    """
        The function creates the label matrix for the loss.
        It is an identity matrix of size [BATCH_SIZE x BATCH_SIZE]
        labels:
            [[1, 0]
             [0, 1]]

        after the proces the labels are smoothed by a small amount to
        account for errors.

        labels:
            [[0.9, 0.1]
             [0.1, 0.9]]


        Pereyra, Gabriel, et al. "Regularizing neural networks by penalizing
        confident output distributions." arXiv preprint arXiv:1701.06548 (2017).
        """
    gold = F.eye(batch_size)
    labels = gold * (1 - self.smoothing_parameter) + (1 - gold) * self.smoothing_parameter / (batch_size - 1)
    return labels