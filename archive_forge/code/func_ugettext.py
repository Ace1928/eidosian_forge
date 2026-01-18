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
def ugettext(self, message):
    missing = object()
    tmsg = self._catalog.get(message, missing)
    if tmsg is missing:
        if self._fallback:
            return self._fallback.ugettext(message)
        return six.text_type(message)
    return tmsg