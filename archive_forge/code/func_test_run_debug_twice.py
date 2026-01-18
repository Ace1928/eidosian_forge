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
def test_run_debug_twice(self):
    _ip = get_ipython()
    with tt.fake_input(['c']):
        _ip.run_line_magic('run', '-d %s' % self.fname)
    with tt.fake_input(['c']):
        _ip.run_line_magic('run', '-d %s' % self.fname)