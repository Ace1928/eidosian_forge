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
def test_ofind_prefers_property_to_instance_level_attribute(self):

    class A(object):

        @property
        def foo(self):
            return 'bar'
    a = A()
    a.__dict__['foo'] = 'baz'
    self.assertEqual(a.foo, 'bar')
    found = ip._ofind('a.foo', [('locals', locals())])
    self.assertIs(found.obj, A.foo)