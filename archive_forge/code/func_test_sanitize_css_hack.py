import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_css_hack(self):
    html = HTML(u'<div style="*position:static">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="_margin:-10px">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))