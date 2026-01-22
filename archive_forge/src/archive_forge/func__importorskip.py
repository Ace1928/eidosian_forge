from __future__ import annotations
import importlib
import platform
import string
import warnings
from contextlib import contextmanager, nullcontext
from unittest import mock  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal  # noqa: F401
from packaging.version import Version
from pandas.testing import assert_frame_equal  # noqa: F401
import xarray.testing
from xarray import Dataset
from xarray.core import utils
from xarray.core.duck_array_ops import allclose_or_equiv  # noqa: F401
from xarray.core.indexing import ExplicitlyIndexed
from xarray.core.options import set_options
from xarray.core.variable import IndexVariable
from xarray.testing import (  # noqa: F401
def _importorskip(modname: str, minversion: str | None=None) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            v = getattr(mod, '__version__', '999')
            if Version(v) < Version(minversion):
                raise ImportError('Minimum version not satisfied')
    except ImportError:
        has = False
    reason = f'requires {modname}'
    if minversion is not None:
        reason += f'>={minversion}'
    func = pytest.mark.skipif(not has, reason=reason)
    return (has, func)