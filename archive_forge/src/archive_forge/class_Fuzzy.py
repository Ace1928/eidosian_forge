import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Fuzzy(RegexBase):

    def __init__(self, subpattern, constraints=None):
        RegexBase.__init__(self)
        if constraints is None:
            constraints = {}
        self.subpattern = subpattern
        self.constraints = constraints
        if 'cost' in constraints:
            for e in 'dis':
                if e in constraints['cost']:
                    constraints.setdefault(e, (0, None))
        if set(constraints) & set('dis'):
            for e in 'dis':
                constraints.setdefault(e, (0, 0))
        else:
            for e in 'dis':
                constraints.setdefault(e, (0, None))
        constraints.setdefault('e', (0, None))
        if 'cost' in constraints:
            for e in 'dis':
                constraints['cost'].setdefault(e, 0)
        else:
            constraints['cost'] = {'d': 1, 'i': 1, 's': 1, 'max': constraints['e'][1]}

    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, True)

    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        return self

    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        return self

    def is_atomic(self):
        return self.subpattern.is_atomic()

    def contains_group(self):
        return self.subpattern.contains_group()

    def _compile(self, reverse, fuzzy):
        arguments = []
        for e in 'dise':
            v = self.constraints[e]
            arguments.append(v[0])
            arguments.append(UNLIMITED if v[1] is None else v[1])
        for e in 'dis':
            arguments.append(self.constraints['cost'][e])
        v = self.constraints['cost']['max']
        arguments.append(UNLIMITED if v is None else v)
        flags = 0
        if reverse:
            flags |= REVERSE_OP
        test = self.constraints.get('test')
        if test:
            return [(OP.FUZZY_EXT, flags) + tuple(arguments)] + test.compile(reverse, True) + [(OP.NEXT,)] + self.subpattern.compile(reverse, True) + [(OP.END,)]
        return [(OP.FUZZY, flags) + tuple(arguments)] + self.subpattern.compile(reverse, True) + [(OP.END,)]

    def dump(self, indent, reverse):
        constraints = self._constraints_to_string()
        if constraints:
            constraints = ' ' + constraints
        print('{}FUZZY{}'.format(INDENT * indent, constraints))
        self.subpattern.dump(indent + 1, reverse)

    def is_empty(self):
        return self.subpattern.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and self.subpattern == other.subpattern and (self.constraints == other.constraints)

    def max_width(self):
        return UNLIMITED

    def _constraints_to_string(self):
        constraints = []
        for name in 'ids':
            min, max = self.constraints[name]
            if max == 0:
                continue
            con = ''
            if min > 0:
                con = '{}<='.format(min)
            con += name
            if max is not None:
                con += '<={}'.format(max)
            constraints.append(con)
        cost = []
        for name in 'ids':
            coeff = self.constraints['cost'][name]
            if coeff > 0:
                cost.append('{}{}'.format(coeff, name))
        limit = self.constraints['cost']['max']
        if limit is not None and limit > 0:
            cost = '{}<={}'.format('+'.join(cost), limit)
            constraints.append(cost)
        return ','.join(constraints)