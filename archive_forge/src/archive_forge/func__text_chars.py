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
def _text_chars(self, length, truncate, text, truncate_len):
    """Truncate a string after a certain number of chars."""
    s_len = 0
    end_index = None
    for i, char in enumerate(text):
        if unicodedata.combining(char):
            continue
        s_len += 1
        if end_index is None and s_len > truncate_len:
            end_index = i
        if s_len > length:
            return add_truncation_text(text[:end_index or 0], truncate)
    return text