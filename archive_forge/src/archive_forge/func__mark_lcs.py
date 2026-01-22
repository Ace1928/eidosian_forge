import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _mark_lcs(mask, dirs, m, n):
    while m != 0 and n != 0:
        if dirs[m, n] == '|':
            m -= 1
            n -= 1
            mask[m] = 1
        elif dirs[m, n] == '^':
            m -= 1
        elif dirs[m, n] == '<':
            n -= 1
        else:
            raise UnboundLocalError('Illegal move')
    return mask