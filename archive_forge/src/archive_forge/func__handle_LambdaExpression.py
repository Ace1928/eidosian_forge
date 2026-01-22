import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_LambdaExpression(self, expression, command, x, y):
    variables = DrtTokens.LAMBDA + '%s' % expression.variable + DrtTokens.DOT
    right = self._visit_command(variables, x, y)[0]
    right, bottom = self._handle(expression.term, command, right, y)
    command(variables, x, self._get_centered_top(y, bottom - y, self._get_text_height()))
    return (right, bottom)