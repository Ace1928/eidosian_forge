import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_unicode_range():
    """
    Test that the ranges we test for unicode names give the same number of
    results than testing the full length.
    """
    from IPython.core.completer import _unicode_name_compute, _UNICODE_RANGES
    expected_list = _unicode_name_compute([(0, 1114112)])
    test = _unicode_name_compute(_UNICODE_RANGES)
    len_exp = len(expected_list)
    len_test = len(test)
    message = None
    if len_exp != len_test or len_exp > 131808:
        size, start, stop, prct = recompute_unicode_ranges()
        message = f"_UNICODE_RANGES likely wrong and need updating. This is\n        likely due to a new release of Python. We've find that the biggest gap\n        in unicode characters has reduces in size to be {size} characters\n        ({prct}), from {start}, to {stop}. In completer.py likely update to\n\n            _UNICODE_RANGES = [(32, {start}), ({stop}, 0xe01f0)]\n\n        And update the assertion below to use\n\n            len_exp <= {len_exp}\n        "
    assert len_exp == len_test, message
    assert len_exp <= 143668, message