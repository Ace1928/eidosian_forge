import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def visit_structured(self, function, combinator):
    """:see: Expression.visit_structured()"""
    return combinator(self.variable, function(self.drs))