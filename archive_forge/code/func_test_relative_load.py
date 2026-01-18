import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_relative_load(self):
    loader = DictLoader({'a/1.html': "{% include '2.html' %}", 'a/2.html': "{% include '../b/3.html' %}", 'b/3.html': 'ok'})
    self.assertEqual(loader.load('a/1.html').generate(), b'ok')