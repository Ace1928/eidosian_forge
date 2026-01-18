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
def test_translate_context_with_msg(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <p i18n:ctxt="foo" i18n:msg="num">\n          Foo <span>There is ${num} bar</span> Bar\n        </p>\n        </html>')
    translations = DummyTranslations({('foo', 'Foo [1:There is %(num)s bar] Bar'): 'Voh [1:Hay %(num)s barre] Barre'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n        <p>Voh <span>Hay 1 barre</span> Barre</p>\n        </html>', tmpl.generate(num=1).render())