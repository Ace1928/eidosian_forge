import string
from ..sage_helper import _within_sage, sage_method
def alexander_polynomial_group(G):
    phi = MapToGroupRingOfFreeAbelianization(G)
    return alexander_polynomial_basic(G, phi)