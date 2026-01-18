import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_whitespace_directive(self):
    loader = DictLoader({'foo.html': '{% whitespace oneline %}\n    {% for i in range(3) %}\n        {{ i }}\n    {% end %}\n{% whitespace all %}\n    pre\tformatted\n'})
    self.assertEqual(loader.load('foo.html').generate(), b'  0  1  2  \n    pre\tformatted\n')