import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_src_javascript(self):
    html = HTML(u'<img src=\'javascript:alert("foo")\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    html = HTML(u'<IMG SRC=\'JaVaScRiPt:alert("foo")\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    src = u'<IMG SRC=`javascript:alert("RSnake says, \'foo\'")`>'
    self.assert_parse_error_or_equal('<img/>', src)
    html = HTML(u'<IMG SRC=\'&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;alert("foo")\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    html = HTML(u'<IMG SRC=\'&#0000106&#0000097&#0000118&#0000097&#0000115&#0000099&#0000114&#0000105&#0000112&#0000116&#0000058alert("foo")\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    html = HTML(u'<IMG SRC=\'&#x6A&#x61&#x76&#x61&#x73&#x63&#x72&#x69&#x70&#x74&#x3A;alert("foo")\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    html = HTML(u'<IMG SRC=\'jav\tascript:alert("foo");\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
    html = HTML(u'<IMG SRC=\'jav&#x09;ascript:alert("foo");\'>')
    self.assertEqual('<img/>', (html | HTMLSanitizer()).render())