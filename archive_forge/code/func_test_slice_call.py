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
def test_slice_call(self):

    def f():
        return 2
    suite = Suite('x = numbers[f()]')
    data = {'numbers': [0, 1, 2, 3, 4], 'f': f}
    suite.execute(data)
    self.assertEqual(2, data['x'])