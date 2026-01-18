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
def test_ofind_line_magic(self):
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def lmagic(line):
        """A line magic"""
    lfind = ip._ofind('lmagic')
    info = OInfo(found=True, isalias=False, ismagic=True, namespace='IPython internal', obj=lmagic, parent=None)
    self.assertEqual(lfind, info)