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
def test_text_noescape_quotes(self):
    """
        Verify that outputting context data in text nodes doesn't escape
        quotes.
        """
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          $myvar\n        </div>')
    self.assertEqual('<div>\n          "foo"\n        </div>', str(tmpl.generate(myvar='"foo"')))