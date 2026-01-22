import contextlib
from functools import partial
from unittest import TestCase
from unittest.util import safe_repr
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..core import (
from ..core.options import Cycle, Options
from ..core.util import cast_array_to_int64, datetime_types, dt_to_int, is_float
from . import *  # noqa (All Elements need to support comparison)
class ComparisonInterface:
    """
    This class is designed to allow equality testing to work
    seamlessly with unittest.TestCase as a mix-in by implementing a
    compatible interface (namely the assertEqual method).

    The assertEqual class method is to be overridden by an instance
    method of the same name when used as a mix-in with TestCase. The
    contents of the equality_type_funcs dictionary is suitable for use
    with TestCase.addTypeEqualityFunc.
    """
    equality_type_funcs = {}
    failureException = AssertionError

    @classmethod
    def simple_equality(cls, first, second, msg=None):
        """
        Classmethod equivalent to unittest.TestCase method (longMessage = False.)
        """
        check = first == second
        if not isinstance(check, bool) and hasattr(check, 'all'):
            check = check.all()
        if not check:
            standardMsg = f'{safe_repr(first)} != {safe_repr(second)}'
            raise cls.failureException(msg or standardMsg)

    @classmethod
    def assertEqual(cls, first, second, msg=None):
        """
        Classmethod equivalent to unittest.TestCase method
        """
        asserter = None
        if type(first) is type(second) or (is_float(first) and is_float(second)):
            asserter = cls.equality_type_funcs.get(type(first))
            if asserter is not None:
                if isinstance(asserter, str):
                    asserter = getattr(cls, asserter)
        if asserter is None:
            asserter = cls.simple_equality
        if msg is None:
            asserter(first, second)
        else:
            asserter(first, second, msg=msg)