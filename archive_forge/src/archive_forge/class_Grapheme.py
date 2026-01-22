import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Grapheme(RegexBase):

    def _compile(self, reverse, fuzzy):
        grapheme_matcher = Atomic(Sequence([LazyRepeat(AnyAll(), 1, None), GraphemeBoundary()]))
        return grapheme_matcher.compile(reverse, fuzzy)

    def dump(self, indent, reverse):
        print('{}GRAPHEME'.format(INDENT * indent))

    def max_width(self):
        return UNLIMITED