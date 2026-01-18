import pyparsing as pp
from operator import mul
from functools import reduce
def verify_result(t):
    if '_skipped' in t:
        t['pass'] = False
    elif 'expected' in t:
        t['pass'] = t.result == t.expected