import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def CreateModule(self, name, file_name=None):
    if file_name is None:
        file_name = '%s.py' % name
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module