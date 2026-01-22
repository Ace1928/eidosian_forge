import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
def _assertPreciseEqual(self, first, second, prec='exact', ulps=1, msg=None, ignore_sign_on_zero=False, abs_tol=None):
    """Recursive workhorse for assertPreciseEqual()."""

    def _assertNumberEqual(first, second, delta=None):
        if delta is None or first == second == 0.0 or math.isinf(first) or math.isinf(second):
            self.assertEqual(first, second, msg=msg)
            if not ignore_sign_on_zero:
                try:
                    if math.copysign(1, first) != math.copysign(1, second):
                        self.fail(self._formatMessage(msg, '%s != %s' % (first, second)))
                except TypeError:
                    pass
        else:
            self.assertAlmostEqual(first, second, delta=delta, msg=msg)
    first_family = self._detect_family(first)
    second_family = self._detect_family(second)
    assertion_message = 'Type Family mismatch. (%s != %s)' % (first_family, second_family)
    if msg:
        assertion_message += ': %s' % (msg,)
    self.assertEqual(first_family, second_family, msg=assertion_message)
    compare_family = first_family
    if compare_family == 'ndarray':
        dtype = self._fix_dtype(first.dtype)
        self.assertEqual(dtype, self._fix_dtype(second.dtype))
        self.assertEqual(first.ndim, second.ndim, 'different number of dimensions')
        self.assertEqual(first.shape, second.shape, 'different shapes')
        self.assertEqual(first.flags.writeable, second.flags.writeable, 'different mutability')
        self.assertEqual(self._fix_strides(first), self._fix_strides(second), 'different strides')
        if first.dtype != dtype:
            first = first.astype(dtype)
        if second.dtype != dtype:
            second = second.astype(dtype)
        for a, b in zip(first.flat, second.flat):
            self._assertPreciseEqual(a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
        return
    elif compare_family == 'sequence':
        self.assertEqual(len(first), len(second), msg=msg)
        for a, b in zip(first, second):
            self._assertPreciseEqual(a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
        return
    elif compare_family == 'exact':
        exact_comparison = True
    elif compare_family in ['complex', 'approximate']:
        exact_comparison = False
    elif compare_family == 'enum':
        self.assertIs(first.__class__, second.__class__)
        self._assertPreciseEqual(first.value, second.value, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
        return
    elif compare_family == 'unknown':
        self.assertIs(first.__class__, second.__class__)
        exact_comparison = True
    else:
        assert 0, 'unexpected family'
    if hasattr(first, 'dtype') and hasattr(second, 'dtype'):
        self.assertEqual(first.dtype, second.dtype)
    if isinstance(first, self._bool_types) != isinstance(second, self._bool_types):
        assertion_message = 'Mismatching return types (%s vs. %s)' % (first.__class__, second.__class__)
        if msg:
            assertion_message += ': %s' % (msg,)
        self.fail(assertion_message)
    try:
        if cmath.isnan(first) and cmath.isnan(second):
            return
    except TypeError:
        pass
    if abs_tol is not None:
        if abs_tol == 'eps':
            rtol = np.finfo(type(first)).eps
        elif isinstance(abs_tol, float):
            rtol = abs_tol
        else:
            raise ValueError('abs_tol is not "eps" or a float, found %s' % abs_tol)
        if abs(first - second) < rtol:
            return
    exact_comparison = exact_comparison or prec == 'exact'
    if not exact_comparison and prec != 'exact':
        if prec == 'single':
            bits = 24
        elif prec == 'double':
            bits = 53
        else:
            raise ValueError('unsupported precision %r' % (prec,))
        k = 2 ** (ulps - bits - 1)
        delta = k * (abs(first) + abs(second))
    else:
        delta = None
    if isinstance(first, self._complex_types):
        _assertNumberEqual(first.real, second.real, delta)
        _assertNumberEqual(first.imag, second.imag, delta)
    elif isinstance(first, (np.timedelta64, np.datetime64)):
        if np.isnat(first):
            self.assertEqual(np.isnat(first), np.isnat(second))
        else:
            _assertNumberEqual(first, second, delta)
    else:
        _assertNumberEqual(first, second, delta)