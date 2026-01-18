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
def test_translate_i18n_choose_as_element_with_attributes(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:choose numeral="num" params="num">\n            <p i18n:singular="" title="Things">\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </p>\n            <p i18n:plural="" title="Things">\n              There are <a href="$link" title="View things">${num} things</a>.\n            </p>\n          </i18n:choose>\n        </html>')
    translations = DummyTranslations({'Things': 'Sachen', 'View thing': 'Sache betrachten', 'View things': 'Sachen betrachten', ('There is [1:%(num)s thing].', 0): 'Da ist [1:%(num)s Sache].', ('There is [1:%(num)s thing].', 1): 'Da sind [1:%(num)s Sachen].'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n            <p title="Sachen">Da ist <a href="/things" title="Sache betrachten">1 Sache</a>.</p>\n        </html>', tmpl.generate(link='/things', num=1).render(encoding=None))
    self.assertEqual(u'<html>\n            <p title="Sachen">Da sind <a href="/things" title="Sachen betrachten">3 Sachen</a>.</p>\n        </html>', tmpl.generate(link='/things', num=3).render(encoding=None))