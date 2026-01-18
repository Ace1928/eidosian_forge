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
def test_translate_i18n_domain_nested_directives(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">Bar</p>\n          <div i18n:domain="foo">\n            <p i18n:msg="">FooBar</p>\n            <p i18n:domain="bar" i18n:msg="">Bar</p>\n            <p>Bar</p>\n          </div>\n          <p>Bar</p>\n        </html>')
    translations = DummyTranslations({'Bar': 'Voh'})
    translations.add_domain('foo', {'FooBar': 'BarFoo', 'Bar': 'foo_Bar'})
    translations.add_domain('bar', {'Bar': 'bar_Bar'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>Voh</p>\n          <div>\n            <p>BarFoo</p>\n            <p>bar_Bar</p>\n            <p>foo_Bar</p>\n          </div>\n          <p>Voh</p>\n        </html>', tmpl.generate().render())