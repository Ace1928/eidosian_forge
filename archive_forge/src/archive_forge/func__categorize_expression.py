from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _categorize_expression(self, current):
    if isinstance(current, NegatedExpression):
        return self._categorize_NegatedExpression(current)
    elif isinstance(current, FunctionVariableExpression):
        return Categories.PROP
    elif TableauProver.is_atom(current):
        return Categories.ATOM
    elif isinstance(current, AllExpression):
        return Categories.ALL
    elif isinstance(current, AndExpression):
        return Categories.AND
    elif isinstance(current, OrExpression):
        return Categories.OR
    elif isinstance(current, ImpExpression):
        return Categories.IMP
    elif isinstance(current, IffExpression):
        return Categories.IFF
    elif isinstance(current, EqualityExpression):
        return Categories.EQ
    elif isinstance(current, ExistsExpression):
        return Categories.EXISTS
    elif isinstance(current, ApplicationExpression):
        return Categories.APP
    else:
        raise ProverParseError('cannot categorize %s' % current.__class__.__name__)