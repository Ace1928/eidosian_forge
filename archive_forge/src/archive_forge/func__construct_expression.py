from math import prod
from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public
def _construct_expression(coeffs, opt):
    """The last resort case, i.e. use the expression domain. """
    domain, result = (EX, [])
    for coeff in coeffs:
        result.append(domain.from_sympy(coeff))
    return (domain, result)