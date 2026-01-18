import numpy as np
from numpy import pi, sin, cos, arccos, sqrt, dot
from numpy.linalg import norm
from ase.cell import Cell  # noqa
def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y / norm(y)