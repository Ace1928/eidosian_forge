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
def test_run__name__():
    with TemporaryDirectory() as td:
        path = pjoin(td, 'foo.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('q = __name__')
        _ip.user_ns.pop('q', None)
        _ip.run_line_magic('run', '{}'.format(path))
        assert _ip.user_ns.pop('q') == '__main__'
        _ip.run_line_magic('run', '-n {}'.format(path))
        assert _ip.user_ns.pop('q') == 'foo'
        try:
            _ip.run_line_magic('run', '-i -n {}'.format(path))
            assert _ip.user_ns.pop('q') == 'foo'
        finally:
            _ip.run_line_magic('reset', '-f')