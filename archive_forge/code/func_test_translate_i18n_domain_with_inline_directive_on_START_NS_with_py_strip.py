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
def test_translate_i18n_domain_with_inline_directive_on_START_NS_with_py_strip(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n"\n            i18n:domain="foo" py:strip="">\n          <p i18n:msg="">FooBar</p>\n        </html>')
    translations = DummyTranslations({'Bar': 'Voh'})
    translations.add_domain('foo', {'FooBar': 'BarFoo'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('\n          <p>BarFoo</p>\n        ', tmpl.generate().render())