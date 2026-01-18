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
def test_simpledef(self):
    """Test that simple class definitions work."""
    src = 'class foo: pass\ndef f(): return foo()'
    self.mktmp(src)
    _ip.run_line_magic('run', str(self.fname))
    _ip.run_cell('t = isinstance(f(), foo)')
    assert _ip.user_ns['t'] is True