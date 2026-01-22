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
class TestImportNoDeprecate(tt.TempFileMixin):

    def setUp(self):
        """Make a valid python temp file."""
        self.mktmp('\nimport warnings\ndef wrn():\n    warnings.warn(\n        "I AM  A WARNING",\n        DeprecationWarning\n    )\n')
        super().setUp()

    def test_no_dep(self):
        """
        No deprecation warning should be raised from imported functions
        """
        ip.run_cell('from {} import wrn'.format(self.fname))
        with tt.AssertNotPrints('I AM  A WARNING'):
            ip.run_cell('wrn()')
        ip.run_cell('del wrn')