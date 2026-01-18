from sympy.polys.rings import ring
from sympy.polys.domains import QQ
from sympy.polys.groebnertools import groebner
def time_vertex_color_12_vertices_24_edges():
    assert groebner(F_2, R) == [1]