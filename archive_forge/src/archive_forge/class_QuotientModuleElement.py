from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class QuotientModuleElement(ModuleElement):
    """Element of a quotient module."""

    def eq(self, d1, d2):
        """Equality comparison."""
        return self.module.killed_module.contains(d1 - d2)

    def __repr__(self):
        return repr(self.data) + ' + ' + repr(self.module.killed_module)