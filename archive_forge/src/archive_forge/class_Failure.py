import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Failure(ZeroWidthBase):
    _op_name = 'FAILURE'

    def _compile(self, reverse, fuzzy):
        return [(OP.FAILURE,)]