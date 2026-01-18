import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
def test_parsekeywordpairs(self):

    def fails(spam=['-a', '-b']):
        pass
    argspec = inspection.getfuncprops('fails', fails)
    self.assertIsNotNone(argspec)
    defaults = argspec.argspec.defaults
    self.assertEqual(str(defaults[0]), '["-a", "-b"]')