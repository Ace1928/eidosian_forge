import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class BareQuotedString(QuotedString):
    token_type = 'bare-quoted-string'

    def __str__(self):
        return quote_string(''.join((str(x) for x in self)))

    @property
    def value(self):
        return ''.join((str(x) for x in self))