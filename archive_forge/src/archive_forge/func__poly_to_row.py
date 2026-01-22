from .polynomial import Polynomial, Monomial
from . import matrix
def _poly_to_row(poly, base_var, base_degree, extension_var, extension_degree):
    row = base_degree * extension_degree * [0]
    for m in poly.get_monomials():
        degrees = dict(m.get_vars())
        degree1 = degrees.get(base_var, 0)
        degree2 = degrees.get(extension_var, 0)
        index = degree2 * base_degree + degree1
        row[index] = m.get_coefficient()
    return row