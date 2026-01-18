import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
def test_unwrap_element(self):
    self.assertEqual(self._unwrap('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (INSIDE, TEXT, u'FOO'), (None, END, u'root')])