import pyglet.gl as pgl
from sympy.core import S
def vec_subs(a, b):
    return tuple((a[i] - b[i] for i in range(len(a))))