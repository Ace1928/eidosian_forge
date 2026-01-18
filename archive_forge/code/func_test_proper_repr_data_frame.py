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
def test_proper_repr_data_frame():
    df = pd.DataFrame(index=[1, 2, 3], data=[[11, 21.0], [12, 22.0], [13, 23.0]], columns=['a', 'b'])
    df2 = eval(proper_repr(df))
    assert df2['a'].dtype == np.int64
    assert df2['b'].dtype == float
    pd.testing.assert_frame_equal(df2, df)
    df = pd.DataFrame(index=pd.Index([1, 2, 3], name='test'), data=[[11, 21.0], [12, 22.0], [13, 23.0]], columns=['a', 'b'])
    df2 = eval(proper_repr(df))
    pd.testing.assert_frame_equal(df2, df)
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)], names=['x', 'y']), data=[[11, 21.0], [12, 22.0], [13, 23.0]], columns=pd.Index(['a', 'b'], name='c'))
    df2 = eval(proper_repr(df))
    pd.testing.assert_frame_equal(df2, df)