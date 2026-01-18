import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_raw_expression(self):
    loader = DictLoader(self.templates)

    def render(name):
        return loader.load(name).generate(name='<>&"')
    self.assertEqual(render('raw_expression.html'), b'expr: &lt;&gt;&amp;&quot;\nraw: <>&"')