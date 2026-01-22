import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _mgu_var(var, expression, bindings):
    if var.variable in expression.free() | expression.constants():
        raise BindingException((var, expression))
    else:
        return BindingDict([(var.variable, expression)]) + bindings