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
def test_ofind_multiple_attribute_lookups(self):

    class A(object):

        @property
        def foo(self):
            raise NotImplementedError()
    a = A()
    a.a = A()
    a.a.a = A()
    found = ip._ofind('a.a.a.foo', [('locals', locals())])
    info = OInfo(found=True, isalias=False, ismagic=False, namespace='locals', obj=A.foo, parent=a.a.a)
    self.assertEqual(found, info)