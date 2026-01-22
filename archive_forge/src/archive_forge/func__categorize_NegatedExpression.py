from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _categorize_NegatedExpression(self, current):
    negated = current.term
    if isinstance(negated, NegatedExpression):
        return Categories.D_NEG
    elif isinstance(negated, FunctionVariableExpression):
        return Categories.N_PROP
    elif TableauProver.is_atom(negated):
        return Categories.N_ATOM
    elif isinstance(negated, AllExpression):
        return Categories.N_ALL
    elif isinstance(negated, AndExpression):
        return Categories.N_AND
    elif isinstance(negated, OrExpression):
        return Categories.N_OR
    elif isinstance(negated, ImpExpression):
        return Categories.N_IMP
    elif isinstance(negated, IffExpression):
        return Categories.N_IFF
    elif isinstance(negated, EqualityExpression):
        return Categories.N_EQ
    elif isinstance(negated, ExistsExpression):
        return Categories.N_EXISTS
    elif isinstance(negated, ApplicationExpression):
        return Categories.N_APP
    else:
        raise ProverParseError('cannot categorize %s' % negated.__class__.__name__)