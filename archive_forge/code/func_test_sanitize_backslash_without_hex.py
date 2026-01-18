import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_backslash_without_hex(self):
    html = HTML(u'<div style="top:e\\xp\\ression(alert())">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    input_str = u'<div style="top:e\\\\xp\\\\ression(alert())">XSS</div>'
    html = HTML(input_str)
    self.assertEqual(input_str, six.text_type(html | StyleSanitizer()))