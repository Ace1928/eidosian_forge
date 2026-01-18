import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_unescape_for_display_utf8(self):

    def test(expected, url, encoding='utf-8'):
        disp_url = urlutils.unescape_for_display(url, encoding=encoding)
        self.assertIsInstance(disp_url, str)
        self.assertEqual(expected, disp_url)
    test('http://foo', 'http://foo')
    if sys.platform == 'win32':
        test('C:/foo/path', 'file:///C|/foo/path')
        test('C:/foo/path', 'file:///C:/foo/path')
    else:
        test('/foo/path', 'file:///foo/path')
    test('http://foo/%2Fbaz', 'http://foo/%2Fbaz')
    test('http://host/räksmörgås', 'http://host/r%C3%A4ksm%C3%B6rg%C3%A5s')
    test('http://host/%3B%2F%3F%3A%40%26%3D%2B%24%2C%23', 'http://host/%3B%2F%3F%3A%40%26%3D%2B%24%2C%23')
    test('http://host/%EE%EE%EE/räksmörgås', 'http://host/%EE%EE%EE/r%C3%A4ksm%C3%B6rg%C3%A5s')
    test('http://host/%EE%EE%EE/räksmörgås', 'http://host/%EE%EE%EE/r%C3%A4ksm%C3%B6rg%C3%A5s', encoding='iso-8859-1')
    test('http://host/جوجو', 'http://host/%d8%ac%d9%88%d8%ac%d9%88', encoding='utf-8')
    test('http://host/%d8%ac%d9%88%d8%ac%d9%88', 'http://host/%d8%ac%d9%88%d8%ac%d9%88', encoding='iso-8859-1')