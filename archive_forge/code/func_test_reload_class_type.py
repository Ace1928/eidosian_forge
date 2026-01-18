import os
import platform
import pytest
import sys
import tempfile
import textwrap
import shutil
import random
import time
import traceback
from io import StringIO
from dataclasses import dataclass
import IPython.testing.tools as tt
from unittest import TestCase
from IPython.extensions.autoreload import AutoreloadMagics
from IPython.core.events import EventManager, pre_run_cell
from IPython.testing.decorators import skipif_not_numpy
from IPython.core.interactiveshell import ExecutionInfo
def test_reload_class_type(self):
    self.shell.magic_autoreload('2')
    mod_name, mod_fn = self.new_module('\n            class Test():\n                def meth(self):\n                    return "old"\n        ')
    assert 'test' not in self.shell.ns
    assert 'result' not in self.shell.ns
    self.shell.run_code('from %s import Test' % mod_name)
    self.shell.run_code('test = Test()')
    self.write_file(mod_fn, '\n            class Test():\n                def meth(self):\n                    return "new"\n        ')
    test_object = self.shell.ns['test']
    self.shell.run_code('pass')
    test_class = pickle_get_current_class(test_object)
    assert isinstance(test_object, test_class)
    self.shell.run_code('import pickle')
    self.shell.run_code('p = pickle.dumps(test)')