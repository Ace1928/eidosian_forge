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
def test_unaryop_not(self):
    self.assertEqual(False, Expression('not True').evaluate({}))
    self.assertEqual(False, Expression('not x').evaluate({'x': True}))