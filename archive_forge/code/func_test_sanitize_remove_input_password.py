import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_input_password(self):
    html = HTML(u'<form><input type="password" /></form>')
    self.assertEqual('<form/>', (html | HTMLSanitizer()).render())