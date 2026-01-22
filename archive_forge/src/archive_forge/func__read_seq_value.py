import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _read_seq_value(self, s, position, reentrances, match, close_paren, seq_class, plus_class):
    """
        Helper function used by read_tuple_value and read_set_value.
        """
    cp = re.escape(close_paren)
    position = match.end()
    m = re.compile('\\s*/?\\s*%s' % cp).match(s, position)
    if m:
        return (seq_class(), m.end())
    values = []
    seen_plus = False
    while True:
        m = re.compile('\\s*%s' % cp).match(s, position)
        if m:
            if seen_plus:
                return (plus_class(values), m.end())
            else:
                return (seq_class(values), m.end())
        val, position = self.read_value(s, position, reentrances)
        values.append(val)
        m = re.compile('\\s*(,|\\+|(?=%s))\\s*' % cp).match(s, position)
        if not m:
            raise ValueError("',' or '+' or '%s'" % cp, position)
        if m.group(1) == '+':
            seen_plus = True
        position = m.end()