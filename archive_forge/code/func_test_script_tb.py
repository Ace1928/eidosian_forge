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
def test_script_tb():
    """Test traceback offset in `ipython script.py`"""
    with TemporaryDirectory() as td:
        path = pjoin(td, 'foo.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(['def foo():', '    return bar()', 'def bar():', "    raise RuntimeError('hello!')", 'foo()']))
        out, err = tt.ipexec(path)
        assert 'execfile' not in out
        assert 'RuntimeError' in out
        assert out.count('---->') == 3