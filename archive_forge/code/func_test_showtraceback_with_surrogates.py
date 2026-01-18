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
@mock.patch('builtins.print')
def test_showtraceback_with_surrogates(self, mocked_print):
    values = []

    def mock_print_func(value, sep=' ', end='\n', file=sys.stdout, flush=False):
        values.append(value)
        if value == chr(55551):
            raise UnicodeEncodeError('utf-8', chr(55551), 0, 1, '')
    mocked_print.side_effect = mock_print_func
    interactiveshell.InteractiveShell._showtraceback(ip, None, None, chr(55551))
    self.assertEqual(mocked_print.call_count, 2)
    self.assertEqual(values, [chr(55551), '\\ud8ff'])