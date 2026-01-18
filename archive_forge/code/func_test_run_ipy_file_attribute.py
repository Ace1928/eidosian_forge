import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
def test_run_ipy_file_attribute(self):
    """Test handling of `__file__` attribute in `%run <file.ipy>`."""
    src = 't = __file__\n'
    self.mktmp(src, ext='.ipy')
    _missing = object()
    file1 = _ip.user_ns.get('__file__', _missing)
    _ip.run_line_magic('run', self.fname)
    file2 = _ip.user_ns.get('__file__', _missing)
    assert _ip.user_ns['t'] == self.fname
    assert file1 == file2