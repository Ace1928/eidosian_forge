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
def test_wrap_root(self):
    self.assertEqual(self._wrap('.'), [(None, START, u'wrap'), (ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (EXIT, END, u'root'), (None, END, u'wrap')])