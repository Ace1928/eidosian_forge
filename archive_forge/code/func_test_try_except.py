import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_try_except(self):
    suite = Suite('try:\n    import somemod\nexcept ImportError:\n    somemod = None\nelse:\n    somemod.dosth()')
    data = {}
    suite.execute(data)
    self.assertEqual(None, data['somemod'])