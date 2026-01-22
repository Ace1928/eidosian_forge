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
class TestAstTransformInputRejection(unittest.TestCase):

    def setUp(self):
        self.transformer = StringRejector()
        ip.ast_transformers.append(self.transformer)

    def tearDown(self):
        ip.ast_transformers.remove(self.transformer)

    def test_input_rejection(self):
        """Check that NodeTransformers can reject input."""
        expect_exception_tb = tt.AssertPrints('InputRejected: test')
        expect_no_cell_output = tt.AssertNotPrints("'unsafe'", suppress=False)
        with expect_exception_tb, expect_no_cell_output:
            ip.run_cell("'unsafe'")
        with expect_exception_tb, expect_no_cell_output:
            res = ip.run_cell("'unsafe'")
        self.assertIsInstance(res.error_before_exec, InputRejected)