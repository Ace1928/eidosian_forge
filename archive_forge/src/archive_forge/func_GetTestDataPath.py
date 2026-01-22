import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def GetTestDataPath(*path):
    return os.path.join(os.path.dirname(__file__), 'testdata', *path)