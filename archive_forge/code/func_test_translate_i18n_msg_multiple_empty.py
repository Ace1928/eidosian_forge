from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
def test_translate_i18n_msg_multiple_empty(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" /> entries per page, starting at page <input type="text" name="num" />.\n          </p>\n        </html>', encoding='utf-8')
    gettext = lambda s: u'[1:] Einträge pro Seite, beginnend auf Seite [2:].'
    translator = Translator(gettext)
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n          <p><input type="text" name="num"/> Einträge pro Seite, beginnend auf Seite <input type="text" name="num"/>.</p>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))