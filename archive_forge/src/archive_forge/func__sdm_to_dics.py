from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable
def _sdm_to_dics(self, s, n):
    """Helper for _sdm_to_vector."""
    from sympy.polys.distributedmodules import sdm_to_dict
    dic = sdm_to_dict(s)
    res = [{} for _ in range(n)]
    for k, v in dic.items():
        res[k[0]][k[1:]] = v
    return res