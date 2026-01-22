import sys
import unittest
from importlib import import_module
from Cryptodome.Util.py3compat import StringIO
class SelfTestError(Exception):

    def __init__(self, message, result):
        Exception.__init__(self, message, result)
        self.message = message
        self.result = result