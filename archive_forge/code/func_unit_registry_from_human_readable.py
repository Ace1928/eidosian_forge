from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def unit_registry_from_human_readable(unit_registry):
    """Deserialization of unit_registry."""
    if unit_registry is None:
        return None
    new_registry = {}
    for k in SI_base_registry:
        factor, u_symbol = unit_registry[k]
        if u_symbol == 1:
            unit_quants = [1]
        else:
            unit_quants = list(pq.Quantity(0, u_symbol).dimensionality.keys())
        if len(unit_quants) != 1:
            raise TypeError('Unknown UnitQuantity: {}'.format(unit_registry[k]))
        else:
            new_registry[k] = factor * unit_quants[0]
    return new_registry