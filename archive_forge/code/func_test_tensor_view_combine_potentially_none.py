from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_tensor_view_combine_potentially_none(self, backend):
    view = backend.get_empty_view()
    assert view.combine_potentially_none(None, None) is None
    a = {'a': [1]}
    b = {'b': [2]}
    assert view.combine_potentially_none(a, None) == a
    assert view.combine_potentially_none(None, a) == a
    assert view.combine_potentially_none(a, b) == view.add_dicts(a, b)