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
def test_replace_with_callback(self):
    count = [0]

    def content():
        count[0] += 1
        yield ('%2i.' % count[0])
    self.assertEqual(self._apply('*', content), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, TEXT, u' 1.'), (None, TEXT, u' 2.'), (None, END, u'root')])