import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_whitespace_by_loader(self):
    templates = {'foo.html': '\t\tfoo\n\n', 'bar.txt': '\t\tbar\n\n'}
    loader = DictLoader(templates, whitespace='all')
    self.assertEqual(loader.load('foo.html').generate(), b'\t\tfoo\n\n')
    self.assertEqual(loader.load('bar.txt').generate(), b'\t\tbar\n\n')
    loader = DictLoader(templates, whitespace='single')
    self.assertEqual(loader.load('foo.html').generate(), b' foo\n')
    self.assertEqual(loader.load('bar.txt').generate(), b' bar\n')
    loader = DictLoader(templates, whitespace='oneline')
    self.assertEqual(loader.load('foo.html').generate(), b' foo ')
    self.assertEqual(loader.load('bar.txt').generate(), b' bar ')