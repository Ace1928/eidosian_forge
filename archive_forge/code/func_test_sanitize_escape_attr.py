import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_escape_attr(self):
    html = HTML(u'<div title="&lt;foo&gt;"></div>')
    self.assertEqual('<div title="&lt;foo&gt;"/>', (html | HTMLSanitizer()).render())