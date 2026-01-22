import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class DotAtomText(TokenList):
    token_type = 'dot-atom-text'
    as_ew_allowed = True