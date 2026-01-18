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
def test_with_in_match(self):
    xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/p">\n            <h1>${select(\'text()\')}</h1>\n            ${select(\'.\')}\n          </py:match>\n          <body><p py:with="foo=\'bar\'">${foo}</p></body>\n        </html>'
    tmpl = MarkupTemplate(xml, filename='test.html')
    self.assertEqual('<html>\n          <body>\n            <h1>bar</h1>\n            <p>bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))