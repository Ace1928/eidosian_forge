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
def test_set_custom_completer():
    num_completers = len(ip.Completer.matchers)

    def foo(*args, **kwargs):
        return "I'm a completer!"
    ip.set_custom_completer(foo, 0)
    assert len(ip.Completer.matchers) == num_completers + 1
    assert ip.Completer.matchers[0]() == "I'm a completer!"
    ip.Completer.custom_matchers.pop()