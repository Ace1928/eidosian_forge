import numpy as np
from ase.optimize.optimize import Dynamics
def update_previous_energies(self, energy):
    """Updates the energy history in self.previous_energies to include the
         current energy."""
    self.previous_energies = np.roll(self.previous_energies, 1)
    self.previous_energies[0] = energy