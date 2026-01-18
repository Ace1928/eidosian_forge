import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_striptags_mid(self):
    markup = Markup('<a href="#">fo<br />o</a>').striptags()
    assert type(markup) is Markup
    self.assertEqual('foo', markup)