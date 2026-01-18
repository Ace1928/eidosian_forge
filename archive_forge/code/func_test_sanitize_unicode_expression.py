import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_unicode_expression(self):
    html = HTML(u'<div style="top:ｅｘｐｒｅｓｓｉｏｎ(alert())">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="top:ＥＸＰＲＥＳＳＩＯＮ(alert())">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="top:expʀessɪoɴ(alert())">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))