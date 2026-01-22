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
def assertPreciseEqual(self, first, second, prec='exact', ulps=1, msg=None, ignore_sign_on_zero=False, abs_tol=None):
    """
        Versatile equality testing function with more built-in checks than
        standard assertEqual().

        For arrays, test that layout, dtype, shape are identical, and
        recursively call assertPreciseEqual() on the contents.

        For other sequences, recursively call assertPreciseEqual() on
        the contents.

        For scalars, test that two scalars or have similar types and are
        equal up to a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.
        ignore_sign_on_zero can be set to True if zeros are to be considered
        equal regardless of their sign bit.
        abs_tol if this is set to a float value its value is used in the
        following. If, however, this is set to the string "eps" then machine
        precision of the type(first) is used in the following instead. This
        kwarg is used to check if the absolute difference in value between first
        and second is less than the value set, if so the numbers being compared
        are considered equal. (This is to handle small numbers typically of
        magnitude less than machine precision).

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
    try:
        self._assertPreciseEqual(first, second, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
    except AssertionError as exc:
        failure_msg = str(exc)
    else:
        return
    self.fail('when comparing %s and %s: %s' % (first, second, failure_msg))