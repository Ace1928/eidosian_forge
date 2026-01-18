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
def test_translate_i18n_msg_with_two_params(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name, time">\n            Written by ${entry.author} at ${entry.time.strftime(\'%H:%M\')}\n          </p>\n        </html>')
    gettext = lambda s: u'%(name)s schrieb dies um %(time)s'
    translator = Translator(gettext)
    translator.setup(tmpl)
    entry = {'author': 'Jim', 'time': datetime(2008, 4, 1, 14, 30)}
    self.assertEqual('<html>\n          <p>Jim schrieb dies um 14:30</p>\n        </html>', tmpl.generate(entry=entry).render())