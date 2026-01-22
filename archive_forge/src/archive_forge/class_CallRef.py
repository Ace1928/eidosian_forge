import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class CallRef(RegexBase):

    def __init__(self, ref, parsed):
        self.ref = ref
        self.parsed = parsed

    def _compile(self, reverse, fuzzy):
        return [(OP.CALL_REF, self.ref)] + self.parsed._compile(reverse, fuzzy) + [(OP.END,)]