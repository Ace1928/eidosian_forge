import math
from typing import List, Optional
import torch
from torch import Tensor
from .optimizer import (
Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        