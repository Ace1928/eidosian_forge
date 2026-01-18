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
@dec.skip_without('numpy')
def test_struct_array_key_completion(self):
    """Test dict key completion applies to numpy struct arrays"""
    import numpy
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = numpy.array([], dtype=[('hello', 'f'), ('world', 'f')])
    _, matches = complete(line_buffer="d['")
    self.assertIn('hello', matches)
    self.assertIn('world', matches)
    dt = numpy.dtype([('my_head', [('my_dt', '>u4'), ('my_df', '>u4')]), ('my_data', '>f4', 5)])
    x = numpy.zeros(2, dtype=dt)
    ip.user_ns['d'] = x[1]
    _, matches = complete(line_buffer="d['")
    self.assertIn('my_head', matches)
    self.assertIn('my_data', matches)

    def completes_on_nested():
        ip.user_ns['d'] = numpy.zeros(2, dtype=dt)
        _, matches = complete(line_buffer="d[1]['my_head']['")
        self.assertTrue(any(['my_dt' in m for m in matches]))
        self.assertTrue(any(['my_df' in m for m in matches]))
    with greedy_completion():
        completes_on_nested()
    with evaluation_policy('limited'):
        completes_on_nested()
    with evaluation_policy('minimal'):
        with pytest.raises(AssertionError):
            completes_on_nested()