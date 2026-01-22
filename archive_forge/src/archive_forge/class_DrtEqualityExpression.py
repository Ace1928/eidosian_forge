import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtEqualityExpression(DrtBinaryExpression, EqualityExpression):

    def fol(self):
        return EqualityExpression(self.first.fol(), self.second.fol())