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
def test_slice(self):
    suite = Suite('x = numbers[0:2]')
    data = {'numbers': [0, 1, 2, 3]}
    suite.execute(data)
    self.assertEqual([0, 1], data['x'])