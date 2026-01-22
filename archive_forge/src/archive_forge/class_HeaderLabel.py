import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class HeaderLabel(TokenList):
    token_type = 'header-label'
    as_ew_allowed = False