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
def test_translate_i18n_choose_as_directive_with_params(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        <i18n:choose numeral="one" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
    translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>Vohs John Doe</p>\n          <p>Voh John Doe</p>\n        </html>', tmpl.generate(one=1, two=2, fname='John', lname='Doe').render())