import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_signature_error_message(self):

    def _dispatcher():
        return ()

    @array_function_dispatch(_dispatcher)
    def func():
        pass
    try:
        func._implementation(bad_arg=3)
    except TypeError as e:
        expected_exception = e
    try:
        func(bad_arg=3)
        raise AssertionError('must fail')
    except TypeError as exc:
        if exc.args[0].startswith('_dispatcher'):
            pytest.skip('Python version is not using __qualname__ for TypeError formatting.')
        assert exc.args == expected_exception.args