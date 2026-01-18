import pyglet.gl as pgl
from sympy.core import S
def update_bounds(b, v):
    if v is None:
        return
    for axis in range(3):
        b[axis][0] = min([b[axis][0], v[axis]])
        b[axis][1] = max([b[axis][1], v[axis]])