import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def _pp(self, indent=''):
    return ['{}{}/{}({}){}'.format(indent, self.__class__.__name__, self.token_type, super().__repr__(), '' if not self.defects else ' {}'.format(self.defects))]