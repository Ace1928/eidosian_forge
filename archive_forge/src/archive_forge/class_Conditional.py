import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Conditional(RegexBase):

    def __init__(self, info, group, yes_item, no_item, position):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.yes_item = yes_item
        self.no_item = no_item
        self.position = position

    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                if self.group == 'DEFINE':
                    self.group = 0
                else:
                    raise error('unknown group', pattern, self.position)
        if not 0 <= self.group <= self.info.group_count:
            raise error('invalid group reference', pattern, self.position)
        self.yes_item.fix_groups(pattern, reverse, fuzzy)
        self.no_item.fix_groups(pattern, reverse, fuzzy)

    def optimise(self, info, reverse):
        yes_item = self.yes_item.optimise(info, reverse)
        no_item = self.no_item.optimise(info, reverse)
        return Conditional(info, self.group, yes_item, no_item, self.position)

    def pack_characters(self, info):
        self.yes_item = self.yes_item.pack_characters(info)
        self.no_item = self.no_item.pack_characters(info)
        return self

    def remove_captures(self):
        self.yes_item = self.yes_item.remove_captures()
        self.no_item = self.no_item.remove_captures()

    def is_atomic(self):
        return self.yes_item.is_atomic() and self.no_item.is_atomic()

    def can_be_affix(self):
        return self.yes_item.can_be_affix() and self.no_item.can_be_affix()

    def contains_group(self):
        return self.yes_item.contains_group() or self.no_item.contains_group()

    def get_firstset(self, reverse):
        return self.yes_item.get_firstset(reverse) | self.no_item.get_firstset(reverse)

    def _compile(self, reverse, fuzzy):
        code = [(OP.GROUP_EXISTS, self.group)]
        code.extend(self.yes_item.compile(reverse, fuzzy))
        add_code = self.no_item.compile(reverse, fuzzy)
        if add_code:
            code.append((OP.NEXT,))
            code.extend(add_code)
        code.append((OP.END,))
        return code

    def dump(self, indent, reverse):
        print('{}GROUP_EXISTS {}'.format(INDENT * indent, self.group))
        self.yes_item.dump(indent + 1, reverse)
        if not self.no_item.is_empty():
            print('{}OR'.format(INDENT * indent))
            self.no_item.dump(indent + 1, reverse)

    def is_empty(self):
        return self.yes_item.is_empty() and self.no_item.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and (self.group, self.yes_item, self.no_item) == (other.group, other.yes_item, other.no_item)

    def max_width(self):
        return max(self.yes_item.max_width(), self.no_item.max_width())

    def __del__(self):
        self.info = None