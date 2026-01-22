from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
def _in_terms_of_generators(self, elem):
    return [self.ring.convert(x, self.quot.ring) for x in self.quot._in_terms_of_generators(self.container.lift(elem))]