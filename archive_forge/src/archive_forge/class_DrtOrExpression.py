import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtOrExpression(DrtBooleanExpression, OrExpression):

    def fol(self):
        return OrExpression(self.first.fol(), self.second.fol())

    def _pretty_subex(self, subex):
        if isinstance(subex, DrtOrExpression):
            return [line[1:-1] for line in subex._pretty()]
        return DrtBooleanExpression._pretty_subex(self, subex)