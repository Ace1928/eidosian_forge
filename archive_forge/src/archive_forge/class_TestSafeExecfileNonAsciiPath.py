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
@pytest.mark.skipif(sys.implementation.name == 'pypy' and (7, 3, 13) < sys.implementation.version < (7, 3, 16), reason='Unicode issues with scandir on PyPy, see https://github.com/pypy/pypy/issues/4860')
class TestSafeExecfileNonAsciiPath(unittest.TestCase):

    @onlyif_unicode_paths
    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        self.TESTDIR = join(self.BASETESTDIR, u'åäö')
        os.mkdir(self.TESTDIR)
        with open(join(self.TESTDIR, 'åäötestscript.py'), 'w', encoding='utf-8') as sfile:
            sfile.write('pass\n')
        self.oldpath = os.getcwd()
        os.chdir(self.TESTDIR)
        self.fname = u'åäötestscript.py'

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        """Test safe_execfile with non-ascii path
        """
        ip.safe_execfile(self.fname, {}, raise_exceptions=True)