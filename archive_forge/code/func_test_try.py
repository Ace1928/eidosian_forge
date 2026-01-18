import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_try(self):
    template = Template(utf8('{% try %}\ntry{% set y = 1/x %}\n{% except %}-except\n{% else %}-else\n{% finally %}-finally\n{% end %}'))
    self.assertEqual(template.generate(x=1), b'\ntry\n-else\n-finally\n')
    self.assertEqual(template.generate(x=0), b'\ntry-except\n-finally\n')