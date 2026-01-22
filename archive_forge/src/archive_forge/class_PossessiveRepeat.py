import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class PossessiveRepeat(GreedyRepeat):

    def is_atomic(self):
        return True

    def _compile(self, reverse, fuzzy):
        subpattern = self.subpattern.compile(reverse, fuzzy)
        if not subpattern:
            return []
        repeat = [self._opcode, self.min_count]
        if self.max_count is None:
            repeat.append(UNLIMITED)
        else:
            repeat.append(self.max_count)
        return [(OP.ATOMIC,), tuple(repeat)] + subpattern + [(OP.END,), (OP.END,)]

    def dump(self, indent, reverse):
        print('{}ATOMIC'.format(INDENT * indent))
        if self.max_count is None:
            limit = 'INF'
        else:
            limit = self.max_count
        print('{}{} {} {}'.format(INDENT * (indent + 1), self._op_name, self.min_count, limit))
        self.subpattern.dump(indent + 2, reverse)