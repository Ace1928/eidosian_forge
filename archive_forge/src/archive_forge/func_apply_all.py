from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def apply_all(self, func: Callable) -> None:
    """
        Apply 'func' across all variables and parameter slices.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
    self.tensor = {var_id: {k: func(v, self.param_to_size[k]) for k, v in parameter_repr.items()} for var_id, parameter_repr in self.tensor.items()}