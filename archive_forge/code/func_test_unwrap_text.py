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
def test_unwrap_text(self):
    self.assertEqual(self._unwrap('foo/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])