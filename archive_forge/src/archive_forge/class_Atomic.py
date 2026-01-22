import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Atomic(RegexBase):

    def __init__(self, subpattern):
        RegexBase.__init__(self)
        self.subpattern = subpattern

    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)

    def optimise(self, info, reverse):
        self.subpattern = self.subpattern.optimise(info, reverse)
        if self.subpattern.is_empty():
            return self.subpattern
        return self

    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self

    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self

    def can_be_affix(self):
        return self.subpattern.can_be_affix()

    def contains_group(self):
        return self.subpattern.contains_group()

    def get_firstset(self, reverse):
        return self.subpattern.get_firstset(reverse)

    def has_simple_start(self):
        return self.subpattern.has_simple_start()

    def _compile(self, reverse, fuzzy):
        return [(OP.ATOMIC,)] + self.subpattern.compile(reverse, fuzzy) + [(OP.END,)]

    def dump(self, indent, reverse):
        print('{}ATOMIC'.format(INDENT * indent))
        self.subpattern.dump(indent + 1, reverse)

    def is_empty(self):
        return self.subpattern.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and self.subpattern == other.subpattern

    def max_width(self):
        return self.subpattern.max_width()

    def get_required_string(self, reverse):
        return self.subpattern.get_required_string(reverse)