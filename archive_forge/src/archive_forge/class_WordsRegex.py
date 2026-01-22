import gzip
import re
import secrets
import unicodedata
from gzip import GzipFile
from gzip import compress as gzip_compress
from io import BytesIO
from django.core.exceptions import SuspiciousFileOperation
from django.utils.functional import SimpleLazyObject, keep_lazy_text, lazy
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy, pgettext
class WordsRegex:

    @staticmethod
    def search(text, pos):
        partial = re_prt.search(text, pos)
        if partial is None or partial[1] is not None:
            return partial
        end = text.find('>', partial.end(0))
        if end < 0:
            return re_notag.search(text, pos + 1)
        else:
            end += 1
            return FakeMatch(text[partial.start(0):end], end)