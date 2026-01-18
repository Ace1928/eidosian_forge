from . import rl
from .core import do_one, exhaust, switch
from .traverse import top_down
def typed(ruletypes):
    """ Apply rules based on the expression type

    inputs:
        ruletypes -- a dict mapping {Type: rule}

    Examples
    ========

    >>> from sympy.strategies import rm_id, typed
    >>> from sympy import Add, Mul
    >>> rm_zeros = rm_id(lambda x: x==0)
    >>> rm_ones  = rm_id(lambda x: x==1)
    >>> remove_idents = typed({Add: rm_zeros, Mul: rm_ones})
    """
    return switch(type, ruletypes)