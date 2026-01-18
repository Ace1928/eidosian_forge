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
def test_normal_comment(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <!-- foo bar -->\n        </div>')
    self.assertEqual('<div>\n          <!-- foo bar -->\n        </div>', str(tmpl.generate()))