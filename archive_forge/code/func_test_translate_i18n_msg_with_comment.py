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
def test_translate_i18n_msg_with_comment(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" i18n:comment="As in foo bar">Foo</p>\n        </html>')
    gettext = lambda s: u'Voh'
    translator = Translator(gettext)
    translator.setup(tmpl)
    self.assertEqual('<html>\n          <p>Voh</p>\n        </html>', tmpl.generate().render())