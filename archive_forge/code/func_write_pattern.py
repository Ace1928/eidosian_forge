from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
def write_pattern(self, filename):
    """ Save calculated data to file specified by ``filename`` string."""
    with open(filename, 'w') as fd:
        self._write_pattern(fd)