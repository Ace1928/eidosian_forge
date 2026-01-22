import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class MIMEVersion(TokenList):
    token_type = 'mime-version'
    major = None
    minor = None