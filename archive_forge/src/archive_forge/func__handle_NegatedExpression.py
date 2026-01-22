import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_NegatedExpression(self, expression, command, x, y):
    right = self._visit_command(DrtTokens.NOT, x, y)[0]
    right, bottom = self._handle(expression.term, command, right, y)
    command(DrtTokens.NOT, x, self._get_centered_top(y, bottom - y, self._get_text_height()))
    return (right, bottom)