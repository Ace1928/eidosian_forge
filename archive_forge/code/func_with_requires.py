import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.18` is installed.

       >>> from cupy import testing
       ... class Test(unittest.TestCase):
       ...     @testing.with_requires('numpy>=1.18')
       ...     def test_for_numpy_1_18(self):
       ...         pass

    Args:
        requirements: A list of string representing requirement condition to
            run a given test case.

    """
    msg = 'requires: {}'.format(','.join(requirements))
    return _skipif(not installed(requirements), reason=msg)