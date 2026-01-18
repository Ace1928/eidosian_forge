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
def test_translate_i18n_msg_with_attributes(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="A helpful paragraph">\n            Please see <a href="help.html" title="Click for help">Help</a>\n          </p>\n        </html>')
    translator = Translator(lambda msgid: {'A helpful paragraph': 'Ein hilfreicher Absatz', 'Click for help': u'Klicken für Hilfe', 'Please see [1:Help]': u'Siehe bitte [1:Hilfe]'}[msgid])
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n          <p title="Ein hilfreicher Absatz">Siehe bitte <a href="help.html" title="Klicken für Hilfe">Hilfe</a></p>\n        </html>', tmpl.generate().render(encoding=None))