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
class TestWarningSuppression(unittest.TestCase):

    def test_warning_suppression(self):
        ip.run_cell('import warnings')
        try:
            with self.assertWarnsRegex(UserWarning, 'asdf'):
                ip.run_cell("warnings.warn('asdf')")
            with self.assertWarnsRegex(UserWarning, 'asdf'):
                ip.run_cell("warnings.warn('asdf')")
        finally:
            ip.run_cell('del warnings')

    def test_deprecation_warning(self):
        ip.run_cell('\nimport warnings\ndef wrn():\n    warnings.warn(\n        "I AM  A WARNING",\n        DeprecationWarning\n    )\n        ')
        try:
            with self.assertWarnsRegex(DeprecationWarning, 'I AM  A WARNING'):
                ip.run_cell('wrn()')
        finally:
            ip.run_cell('del warnings')
            ip.run_cell('del wrn')