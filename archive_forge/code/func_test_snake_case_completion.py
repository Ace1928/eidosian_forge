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
def test_snake_case_completion(self):
    ip = get_ipython()
    ip.Completer.use_jedi = False
    ip.user_ns['some_three'] = 3
    ip.user_ns['some_four'] = 4
    _, matches = ip.complete('s_', 'print(s_f')
    self.assertIn('some_three', matches)
    self.assertIn('some_four', matches)