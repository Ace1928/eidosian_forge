import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def alpha_convert(self, newvar):
    """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
    return self.__class__(newvar, self.term.replace(self.variable, DrtVariableExpression(newvar), True))