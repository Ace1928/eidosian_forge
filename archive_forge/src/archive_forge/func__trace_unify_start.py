import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _trace_unify_start(path, fval1, fval2):
    if path == ():
        print('\nUnification trace:')
    else:
        fullname = '.'.join(('%s' % n for n in path))
        print('  ' + '|   ' * (len(path) - 1) + '|')
        print('  ' + '|   ' * (len(path) - 1) + '| Unify feature: %s' % fullname)
    print('  ' + '|   ' * len(path) + ' / ' + _trace_valrepr(fval1))
    print('  ' + '|   ' * len(path) + '|\\ ' + _trace_valrepr(fval2))