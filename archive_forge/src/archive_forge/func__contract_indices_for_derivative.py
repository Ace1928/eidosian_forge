from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul
@classmethod
def _contract_indices_for_derivative(cls, expr, variables):
    variables_opposite_valence = []
    for i in variables:
        if isinstance(i, Tensor):
            i_free_indices = i.get_free_indices()
            variables_opposite_valence.append(i.xreplace({k: -k for k in i_free_indices}))
        elif isinstance(i, Symbol):
            variables_opposite_valence.append(i)
    args, indices, free, dum = TensMul._tensMul_contract_indices([expr] + variables_opposite_valence, replace_indices=True)
    for i in range(1, len(args)):
        args_i = args[i]
        if isinstance(args_i, Tensor):
            i_indices = args[i].get_free_indices()
            args[i] = args[i].xreplace({k: -k for k in i_indices})
    return (args, indices, free, dum)