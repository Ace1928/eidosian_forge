import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _get_least_common(self, atoms):
    """Returns the least common element in atoms. If more than one,
        returns the first encountered."""
    symbols = [atom.symbol for atom in atoms]
    least = ['', np.inf]
    for element in set(symbols):
        count = symbols.count(element)
        if count < least[1]:
            least = [element, count]
    return least