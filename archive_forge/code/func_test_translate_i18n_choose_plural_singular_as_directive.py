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
def test_translate_i18n_choose_plural_singular_as_directive(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <i18n:singular>FooBar</i18n:singular>\n          <i18n:plural>FooBars</i18n:plural>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <i18n:singular>FooBar</i18n:singular>\n          <i18n:plural>FooBars</i18n:plural>\n        </i18n:choose>\n        </html>')
    translations = DummyTranslations({('FooBar', 0): 'FuBar', ('FooBars', 1): 'FuBars', 'FooBar': 'FuBar', 'FooBars': 'FuBars'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          FuBars\n          FuBar\n        </html>', tmpl.generate(one=1, two=2).render())