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
def apply_to_parameters(self, func: Callable, parameter_representation: dict[int, sp.spmatrix]) -> dict[int, sp.spmatrix]:
    """
        Apply 'func' to each slice of the parameter representation.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
    return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}