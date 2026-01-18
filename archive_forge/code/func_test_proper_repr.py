import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_proper_repr():
    v = sympy.Symbol('t') * 3
    v2 = eval(proper_repr(v))
    assert v2 == v
    v = np.array([1, 2, 3], dtype=np.complex64)
    v2 = eval(proper_repr(v))
    np.testing.assert_array_equal(v2, v)
    assert v2.dtype == v.dtype