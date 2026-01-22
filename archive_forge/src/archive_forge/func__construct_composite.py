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
def _construct_composite(coeffs, opt):
    """Handle composite domains, e.g.: ZZ[X], QQ[X], ZZ(X), QQ(X). """
    numers, denoms = ([], [])
    for coeff in coeffs:
        numer, denom = coeff.as_numer_denom()
        numers.append(numer)
        denoms.append(denom)
    polys, gens = parallel_dict_from_basic(numers + denoms)
    if not gens:
        return None
    if opt.composite is None:
        if any((gen.is_number and gen.is_algebraic for gen in gens)):
            return None
        all_symbols = set()
        for gen in gens:
            symbols = gen.free_symbols
            if all_symbols & symbols:
                return None
            else:
                all_symbols |= symbols
    n = len(gens)
    k = len(polys) // 2
    numers = polys[:k]
    denoms = polys[k:]
    if opt.field:
        fractions = True
    else:
        fractions, zeros = (False, (0,) * n)
        for denom in denoms:
            if len(denom) > 1 or zeros not in denom:
                fractions = True
                break
    coeffs = set()
    if not fractions:
        for numer, denom in zip(numers, denoms):
            denom = denom[zeros]
            for monom, coeff in numer.items():
                coeff /= denom
                coeffs.add(coeff)
                numer[monom] = coeff
    else:
        for numer, denom in zip(numers, denoms):
            coeffs.update(list(numer.values()))
            coeffs.update(list(denom.values()))
    rationals = floats = complexes = False
    float_numbers = []
    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            floats = True
            float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex is not None:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
    max_prec = max((c._prec for c in float_numbers)) if float_numbers else 53
    if floats and complexes:
        ground = ComplexField(prec=max_prec)
    elif floats:
        ground = RealField(prec=max_prec)
    elif complexes:
        if rationals:
            ground = QQ_I
        else:
            ground = ZZ_I
    elif rationals:
        ground = QQ
    else:
        ground = ZZ
    result = []
    if not fractions:
        domain = ground.poly_ring(*gens)
        for numer in numers:
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)
            result.append(domain(numer))
    else:
        domain = ground.frac_field(*gens)
        for numer, denom in zip(numers, denoms):
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)
            for monom, coeff in denom.items():
                denom[monom] = ground.from_sympy(coeff)
            result.append(domain((numer, denom)))
    return (domain, result)