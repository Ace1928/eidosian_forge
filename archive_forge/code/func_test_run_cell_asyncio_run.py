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
def test_run_cell_asyncio_run():
    ip.run_cell('import asyncio')
    result = ip.run_cell('await asyncio.sleep(0.01); 1')
    assert ip.user_ns['_'] == 1
    result = ip.run_cell('asyncio.run(asyncio.sleep(0.01)); 2')
    assert ip.user_ns['_'] == 2
    result = ip.run_cell('await asyncio.sleep(0.01); 3')
    assert ip.user_ns['_'] == 3