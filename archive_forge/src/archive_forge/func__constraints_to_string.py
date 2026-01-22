import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
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