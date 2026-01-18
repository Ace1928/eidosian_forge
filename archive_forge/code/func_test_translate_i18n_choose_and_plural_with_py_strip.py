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
def test_translate_i18n_choose_and_plural_with_py_strip(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="" py:strip="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
    translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <div>\n            Voh John Doe\n          </div>\n        </html>', tmpl.generate(two=1, fname='John', lname='Doe').render())