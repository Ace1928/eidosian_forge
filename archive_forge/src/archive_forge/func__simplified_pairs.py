from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def _simplified_pairs(terms):
    """
    Reduces a set of minterms, if possible, to a simplified set of minterms
    with one less variable in the terms using QM method.
    """
    if not terms:
        return []
    simplified_terms = []
    todo = list(range(len(terms)))
    termdict = defaultdict(list)
    for n, term in enumerate(terms):
        ones = sum([1 for t in term if t == 1])
        termdict[ones].append(n)
    variables = len(terms[0])
    for k in range(variables):
        for i in termdict[k]:
            for j in termdict[k + 1]:
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    todo[i] = todo[j] = None
                    newterm = terms[i][:]
                    newterm[index] = 3
                    if newterm not in simplified_terms:
                        simplified_terms.append(newterm)
    if simplified_terms:
        simplified_terms = _simplified_pairs(simplified_terms)
    simplified_terms.extend([terms[i] for i in todo if i is not None])
    return simplified_terms