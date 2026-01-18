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
def test_should_run_async():
    assert not ip.should_run_async('a = 5', transformed_cell='a = 5')
    assert ip.should_run_async('await x', transformed_cell='await x')
    assert ip.should_run_async('import asyncio; await asyncio.sleep(1)', transformed_cell='import asyncio; await asyncio.sleep(1)')