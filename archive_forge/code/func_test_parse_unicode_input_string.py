import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_parse_unicode_input_string(self):
    """https://github.com/bernii/querystring-parser/issues/15"""
    qs = u'first_name=%D8%B9%D9%84%DB%8C'
    expected = {u'first_name': u'علی'}
    self.assertEqual(parse(qs.encode('ascii')), expected)
    self.assertEqual(parse(qs), expected)