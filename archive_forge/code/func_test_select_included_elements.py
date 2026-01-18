import doctest
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import six
from genshi.compat import BytesIO, StringIO
from genshi.core import Markup
from genshi.filters.i18n import Translator
from genshi.input import XML
from genshi.template.base import BadDirectiveError, TemplateSyntaxError
from genshi.template.loader import TemplateLoader, TemplateNotFound
from genshi.template.markup import MarkupTemplate
def test_select_included_elements(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
        try:
            file1.write('<li>$item</li>')
        finally:
            file1.close()
        file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
        try:
            file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <ul py:match="ul">${select(\'li\')}</ul>\n                  <ul py:with="items=(1, 2, 3)">\n                    <xi:include href="tmpl1.html" py:for="item in items" />\n                  </ul>\n                </html>')
        finally:
            file2.close()
        loader = TemplateLoader([dirname])
        tmpl = loader.load('tmpl2.html')
        self.assertEqual('<html>\n                  <ul><li>1</li><li>2</li><li>3</li></ul>\n                </html>', tmpl.generate().render(encoding=None))
    finally:
        shutil.rmtree(dirname)