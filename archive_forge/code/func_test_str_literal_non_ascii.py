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
def test_str_literal_non_ascii(self):
    expr = Expression(u"u'þ'")
    self.assertEqual(u'þ', expr.evaluate({}))
    expr = Expression("u'þ'")
    self.assertEqual(u'þ', expr.evaluate({}))
    expr = Expression(wrapped_bytes("b'\\xc3\\xbe'"))
    if IS_PYTHON2:
        self.assertEqual(u'þ', expr.evaluate({}))
    else:
        self.assertEqual(u'þ'.encode('utf-8'), expr.evaluate({}))