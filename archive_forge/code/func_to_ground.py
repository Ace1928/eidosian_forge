from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting
def to_ground(f):
    [c] = f.rep.to_list()
    return c