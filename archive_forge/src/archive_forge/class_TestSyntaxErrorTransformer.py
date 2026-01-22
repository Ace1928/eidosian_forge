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
class TestSyntaxErrorTransformer(unittest.TestCase):
    """Check that SyntaxError raised by an input transformer is handled by run_cell()"""

    @staticmethod
    def transformer(lines):
        for line in lines:
            pos = line.find('syntaxerror')
            if pos >= 0:
                e = SyntaxError('input contains "syntaxerror"')
                e.text = line
                e.offset = pos + 1
                raise e
        return lines

    def setUp(self):
        ip.input_transformers_post.append(self.transformer)

    def tearDown(self):
        ip.input_transformers_post.remove(self.transformer)

    def test_syntaxerror_input_transformer(self):
        with tt.AssertPrints('1234'):
            ip.run_cell('1234')
        with tt.AssertPrints('SyntaxError: invalid syntax'):
            ip.run_cell('1 2 3')
        with tt.AssertPrints('SyntaxError: input contains "syntaxerror"'):
            ip.run_cell('2345  # syntaxerror')
        with tt.AssertPrints('3456'):
            ip.run_cell('3456')