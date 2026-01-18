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
def test_run_nb(self):
    """Test %run notebook.ipynb"""
    pytest.importorskip('nbformat')
    from nbformat import v4, writes
    nb = v4.new_notebook(cells=[v4.new_markdown_cell('The Ultimate Question of Everything'), v4.new_code_cell('answer=42')])
    src = writes(nb, version=4)
    self.mktmp(src, ext='.ipynb')
    _ip.run_line_magic('run', self.fname)
    assert _ip.user_ns['answer'] == 42