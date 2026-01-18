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
def test_run_cell_multiline(self):
    """Multi-block, multi-line cells must execute correctly.
        """
    src = '\n'.join(['x=1', 'y=2', 'if 1:', '    x += 1', '    y += 1'])
    res = ip.run_cell(src)
    self.assertEqual(ip.user_ns['x'], 2)
    self.assertEqual(ip.user_ns['y'], 3)
    self.assertEqual(res.success, True)
    self.assertEqual(res.result, None)