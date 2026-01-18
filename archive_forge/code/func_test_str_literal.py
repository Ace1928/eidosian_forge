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
def test_str_literal(self):
    self.assertEqual('foo', Expression('"foo"').evaluate({}))
    self.assertEqual('foo', Expression('"""foo"""').evaluate({}))
    self.assertEqual(u'foo'.encode('utf-8'), Expression(wrapped_bytes("b'foo'")).evaluate({}))
    self.assertEqual('foo', Expression("'''foo'''").evaluate({}))
    self.assertEqual('foo', Expression("u'foo'").evaluate({}))
    self.assertEqual('foo', Expression("r'foo'").evaluate({}))