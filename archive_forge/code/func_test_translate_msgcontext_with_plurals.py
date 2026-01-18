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
def test_translate_msgcontext_with_plurals(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:ctxt name="foo">\n          <p i18n:choose="num; num">\n            <span i18n:singular="">There is ${num} bar</span>\n            <span i18n:plural="">There are ${num} bars</span>\n          </p>\n        </i18n:ctxt>\n        </html>')
    translations = DummyTranslations({('foo', 'There is %(num)s bar', 0): 'Hay %(num)s barre', ('foo', 'There is %(num)s bar', 1): 'Hay %(num)s barres'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>\n            <span>Hay 1 barre</span>\n          </p>\n        </html>', tmpl.generate(num=1).render())
    self.assertEqual('<html>\n          <p>\n            <span>Hay 2 barres</span>\n          </p>\n        </html>', tmpl.generate(num=2).render())