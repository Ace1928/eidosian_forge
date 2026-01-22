from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
def _ranking_reduce(score: Tensor, num_elements: int) -> Tensor:
    return score / num_elements