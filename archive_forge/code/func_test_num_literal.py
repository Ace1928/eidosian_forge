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
def test_num_literal(self):
    self.assertEqual(42, Expression('42').evaluate({}))
    if IS_PYTHON2:
        self.assertEqual(42, Expression('42L').evaluate({}))
    self.assertEqual(0.42, Expression('.42').evaluate({}))
    if IS_PYTHON2:
        self.assertEqual(7, Expression('07').evaluate({}))
    self.assertEqual(242, Expression('0xF2').evaluate({}))
    self.assertEqual(242, Expression('0XF2').evaluate({}))