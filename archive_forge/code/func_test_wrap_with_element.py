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
def test_wrap_with_element(self):
    element = Element('a', href='http://localhost')
    self.assertEqual(_transform('foo', Transformer('.').wrap(element), with_attrs=True), [(None, START, (u'a', {u'href': u'http://localhost'})), (OUTSIDE, TEXT, u'foo'), (None, END, u'a')])