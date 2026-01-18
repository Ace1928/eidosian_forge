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
def test_run_cell_async():
    ip.run_cell('import asyncio')
    coro = ip.run_cell_async('await asyncio.sleep(0.01)\n5')
    assert asyncio.iscoroutine(coro)
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(coro)
    assert isinstance(result, interactiveshell.ExecutionResult)
    assert result.result == 5