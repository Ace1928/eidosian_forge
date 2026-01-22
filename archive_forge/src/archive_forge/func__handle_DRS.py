import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_DRS(self, expression, command, x, y):
    left = x + self.BUFFER
    bottom = y + self.BUFFER
    if expression.refs:
        refs = ' '.join(('%s' % r for r in expression.refs))
    else:
        refs = '     '
    max_right, bottom = command(refs, left, bottom)
    bottom += self.BUFFER * 2
    if expression.conds:
        for cond in expression.conds:
            right, bottom = self._handle(cond, command, left, bottom)
            max_right = max(max_right, right)
            bottom += self.BUFFER
    else:
        bottom += self._get_text_height() + self.BUFFER
    max_right += self.BUFFER
    return command((max_right, bottom), x, y)