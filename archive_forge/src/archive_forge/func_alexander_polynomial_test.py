import sys
from . import links, tangles
def alexander_polynomial_test():
    """
    Export a bunch of Montesinos knots as PD diagrams and
    compute the Alexander polynomials via KnotTheory and
    also Sage.  Make sure they match.
    """
    try:
        sys.path.append('/Users/dunfield/work/python')
        import nsagetools
    except ImportError:
        print("Skipping this test as you're not within Sage and/or are not Nathan.")
        return
    from sage.all import mathematica, PolynomialRing, ZZ
    mathematica.execute('AppendTo[$Path, "/pkgs"]')
    mathematica.execute('<<KnotTheory`')
    mathematica.execute('Off[KnotTheory::loading]')
    mathematica.execute('Off[KnotTheory::credits]')
    mathematica.execute('MyAlex[L_] := CoefficientList[t^100*Alexander[L][t], t]')

    def alex_by_KnotTheory(L):
        p = mathematica.MyAlex(L.PD_code(True)).sage()
        i = next(iter((i for i, c in enumerate(p) if c != 0)))
        R = PolynomialRing(ZZ, 'a')
        return R(p[i:])

    def alex_match(L):
        p1 = alex_by_KnotTheory(L)
        p2 = nsagetools.alexander_polynomial(L.exterior())
        return 0 in [p1 - p2, p1 + p2]
    for name, K in some_knots():
        assert alex_match(K)
    print('Checked Alexander polynomials via KnotTheory for these 167 knots.')