from .polynomial import Polynomial, Monomial
from . import matrix
def _reduced_polynomial(poly, mod_pol, mod_var, mod_degree):

    def degree_of_monomial(m):
        vars = dict(m.get_vars())
        return vars.get(mod_var, 0)

    def reducing_polynomial(m):

        def new_degree(var, expo):
            if var == mod_var:
                return (var, expo - mod_degree)
            else:
                return (var, expo)
        new_degrees = [new_degree(var, expo) for var, expo in m.get_vars()]
        new_degrees_filtered = tuple([(var, expo) for var, expo in new_degrees if expo > 0])
        monomial = Monomial(m.get_coefficient(), new_degrees_filtered)
        return Polynomial((monomial,))
    while True:
        degree = poly.degree(mod_var)
        if degree < mod_degree:
            return poly
        for m in poly.get_monomials():
            if degree_of_monomial(m) == degree:
                poly = poly - reducing_polynomial(m) * mod_pol