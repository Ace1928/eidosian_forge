import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_close_empty_tag(self):
    html = HTML(u'<a href="#">fo<br>o</a>')
    self.assertEqual('<a href="#">fo<br/>o</a>', (html | HTMLSanitizer()).render())