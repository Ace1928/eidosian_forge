import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
@onlyif_unicode_paths
def test_unicode_in_filename():
    """When a file doesn't exist, the exception raised should be safe to call
    str() on - i.e. in Python 2 it must only have ASCII characters.

    https://github.com/ipython/ipython/issues/875
    """
    try:
        path.get_py_filename('fooéè.py')
    except IOError as ex:
        str(ex)