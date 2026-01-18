import sys
from . import links, tangles
def whitehead():
    a, b, c, d, e = crossings = [Crossing(x) for x in 'abcde']
    a[0] = b[3]
    a[1] = d[0]
    a[2] = d[3]
    a[3] = c[0]
    b[0] = c[3]
    b[1] = e[2]
    b[2] = e[1]
    c[1] = d[2]
    c[2] = e[3]
    d[1] = e[0]
    return Link(crossings)