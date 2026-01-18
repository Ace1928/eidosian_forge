import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_whitespace_by_filename(self):
    loader = DictLoader({'foo.html': '   \n\t\n asdf\t   ', 'bar.js': ' \n\n\n\t qwer     ', 'baz.txt': '\t    zxcv\n\n', 'include.html': '  {% include baz.txt %} \n ', 'include.txt': '\t\t{% include foo.html %}    '})
    self.assertEqual(loader.load('foo.html').generate(), b'\nasdf ')
    self.assertEqual(loader.load('bar.js').generate(), b'\nqwer ')
    self.assertEqual(loader.load('baz.txt').generate(), b'\t    zxcv\n\n')
    self.assertEqual(loader.load('include.html').generate(), b' \t    zxcv\n\n\n')
    self.assertEqual(loader.load('include.txt').generate(), b'\t\t\nasdf     ')