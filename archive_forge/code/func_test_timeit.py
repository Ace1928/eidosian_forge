import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
def test_timeit(self):
    called = set()

    def f(x):
        called.add(x)
    ip.push({'f': f})
    with tt.AssertPrints('std. dev. of'):
        ip.run_line_magic('timeit', '-n1 f(1)')
    self.assertEqual(called, {(1,)})
    called.clear()
    with tt.AssertPrints('std. dev. of'):
        ip.run_cell_magic('timeit', '-n1 f(2)', 'f(3)')
    self.assertEqual(called, {(2,), (3,)})