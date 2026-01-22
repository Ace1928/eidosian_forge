from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@dataclass
class linOpHelper:
    """
    Helper class that allows to access properties of linOps without
    needing to create a full linOps instance
    """
    shape: None | tuple[int, ...] = None
    type: None | str = None
    data: None | int | np.ndarray | list[slice] = None
    args: None | list[linOpHelper] = None