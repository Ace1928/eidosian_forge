import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class PrecompiledCode(RegexBase):

    def __init__(self, code):
        self.code = code

    def _compile(self, reverse, fuzzy):
        return [tuple(self.code)]