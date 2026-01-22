import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
class MockedMethodConfig(object):

    def __init__(self, relative_path, path_params):
        self.relative_path = relative_path
        self.path_params = path_params