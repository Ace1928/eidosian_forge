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
@pytest.mark.xfail(platform.python_implementation() == 'PyPy', reason="expecting __del__ call on exit is unreliable and doesn't happen on PyPy")
def test_obj_del(self):
    """Test that object's __del__ methods are called on exit."""
    src = "class A(object):\n    def __del__(self):\n        print('object A deleted')\na = A()\n"
    self.mktmp(src)
    err = None
    tt.ipexec_validate(self.fname, 'object A deleted', err)