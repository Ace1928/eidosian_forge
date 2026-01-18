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
def test_translate_i18n_msg_elt_nonewline(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg>Please see <a href="help.html">Help</a></i18n:msg>\n        </html>')
    gettext = lambda s: u'Für Details siehe bitte [1:Hilfe]'
    translator = Translator(gettext)
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n          Für Details siehe bitte <a href="help.html">Hilfe</a>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))