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
def test_silent_nodisplayhook(self):
    """run_cell(silent=True) doesn't trigger displayhook"""
    d = dict(called=False)
    trap = ip.display_trap
    save_hook = trap.hook

    def failing_hook(*args, **kwargs):
        d['called'] = True
    try:
        trap.hook = failing_hook
        res = ip.run_cell('1', silent=True)
        self.assertFalse(d['called'])
        self.assertIsNone(res.result)
        ip.run_cell('1')
        self.assertTrue(d['called'])
    finally:
        trap.hook = save_hook