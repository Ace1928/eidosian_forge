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
@keep_lazy_text
def unescape_string_literal(s):
    """
    Convert quoted string literals to unquoted strings with escaped quotes and
    backslashes unquoted::

        >>> unescape_string_literal('"abc"')
        'abc'
        >>> unescape_string_literal("'abc'")
        'abc'
        >>> unescape_string_literal('"a \\"bc\\""')
        'a "bc"'
        >>> unescape_string_literal("'\\'ab\\' c'")
        "'ab' c"
    """
    if not s or s[0] not in '"\'' or s[-1] != s[0]:
        raise ValueError('Not a string literal: %r' % s)
    quote = s[0]
    return s[1:-1].replace('\\%s' % quote, quote).replace('\\\\', '\\')