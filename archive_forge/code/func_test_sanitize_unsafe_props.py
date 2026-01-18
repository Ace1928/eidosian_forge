import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_unsafe_props(self):
    html = HTML(u'<div style="POSITION:RELATIVE">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="behavior:url(test.htc)">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="-ms-behavior:url(test.htc) url(#obj)">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="-o-link:\'javascript:alert(1)\';-o-link-source:current">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
    html = HTML(u'<div style="-moz-binding:url(xss.xbl)">XSS</div>')
    self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))