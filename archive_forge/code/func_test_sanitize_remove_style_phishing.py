import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_style_phishing(self):
    sanitizer = StyleSanitizer()
    html = HTML(u'<div style="position:absolute;top:0"></div>')
    self.assertEqual('<div style="top:0"/>', (html | sanitizer).render())
    html = HTML(u'<div style="margin:10px 20px"></div>')
    self.assertEqual('<div style="margin:10px 20px"/>', (html | sanitizer).render())
    html = HTML(u'<div style="margin:-1000px 0 0"></div>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<div style="margin-left:-2000px 0 0"></div>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<div style="margin-left:1em 1em 1em -4000px"></div>')
    self.assertEqual('<div/>', (html | sanitizer).render())