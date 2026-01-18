import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_unicode_input(self):
    text = u'$fooö$bar'
    tmpl = NewTextTemplate(text)
    self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))