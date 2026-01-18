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
def test_translate_i18n_choose_as_directive_singular_and_plural_with_strip(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <p i18n:singular="" py:strip="">FooBar Singular with Strip</p>\n          <p i18n:plural="">FooBars Plural without Strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="two">\n          <p i18n:singular="">FooBar singular without strip</p>\n          <p i18n:plural="" py:strip="">FooBars plural with strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="">FooBar singular without strip</p>\n          <p i18n:plural="" py:strip="">FooBars plural with strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="" py:strip="">FooBar singular with strip</p>\n          <p i18n:plural="">FooBars plural without strip</p>\n        </i18n:choose>\n        </html>')
    translations = DummyTranslations()
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>FooBars Plural without Strip</p>\n          FooBars plural with strip\n          <p>FooBar singular without strip</p>\n          FooBar singular with strip\n        </html>', tmpl.generate(one=1, two=2).render())