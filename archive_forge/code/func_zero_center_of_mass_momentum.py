import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def zero_center_of_mass_momentum(self, verbose=0):
    """Set the center of mass momentum to zero."""
    cm = self.get_center_of_mass_momentum()
    abscm = np.sqrt(np.sum(cm * cm))
    if verbose and abscm > 0.0001:
        self._warning(self.classname + ': Setting the center-of-mass momentum to zero (was %.6g %.6g %.6g)' % tuple(cm))
    self.atoms.set_momenta(self.atoms.get_momenta() - cm / self._getnatoms())