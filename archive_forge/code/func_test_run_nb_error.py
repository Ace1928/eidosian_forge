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
def test_run_nb_error(self):
    """Test %run notebook.ipynb error"""
    pytest.importorskip('nbformat')
    from nbformat import v4, writes
    pytest.raises(Exception, _ip.magic, 'run')
    pytest.raises(Exception, _ip.magic, 'run foobar.ipynb')
    nb = v4.new_notebook(cells=[v4.new_code_cell('0/0')])
    src = writes(nb, version=4)
    self.mktmp(src, ext='.ipynb')
    pytest.raises(Exception, _ip.magic, 'run %s' % self.fname)