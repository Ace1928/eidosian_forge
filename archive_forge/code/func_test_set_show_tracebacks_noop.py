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
def test_set_show_tracebacks_noop(self):
    """Test the case of the client setting showtracebacks to a no op lambda"""
    result = ip.run_cell('\n            import IPython.core.interactiveshell\n            IPython.core.interactiveshell.InteractiveShell.showtraceback = lambda *args, **kwargs: None\n\n            assert False, "This should not raise an exception"\n        ')
    print(result)
    assert result.result is None
    assert isinstance(result.error_in_exec, AssertionError)
    assert str(result.error_in_exec) == 'This should not raise an exception'