import networkx as nx
from networkx.utils import py_random_state
def uniform_int_from_avg(a, m, seed):
    """Pick a random integer with uniform probability.

    Returns a random integer uniformly taken from a distribution with
    minimum value 'a' and average value 'm', X~U(a,b), E[X]=m, X in N where
    b = 2*m - a.

    Notes
    -----
    p = (b-floor(b))/2
    X = X1 + X2; X1~U(a,floor(b)), X2~B(p)
    E[X] = E[X1] + E[X2] = (floor(b)+a)/2 + (b-floor(b))/2 = (b+a)/2 = m
    """
    from math import floor
    assert m >= a
    b = 2 * m - a
    p = (b - floor(b)) / 2
    X1 = round(seed.random() * (floor(b) - a) + a)
    if seed.random() < p:
        X2 = 1
    else:
        X2 = 0
    return X1 + X2