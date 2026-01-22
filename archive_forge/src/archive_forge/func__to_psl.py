from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _to_psl(m):
    return m / m.determinant().sqrt()