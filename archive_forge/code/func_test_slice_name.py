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
def test_slice_name(self):
    suite = Suite('x = numbers[v]')
    data = {'numbers': [0, 1, 2, 3, 4], 'v': 2}
    suite.execute(data)
    self.assertEqual(2, data['x'])