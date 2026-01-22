import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _calculate_q_past_and_future(self):

    def ekin(p, m=self.atoms.get_masses()):
        p2 = np.sum(p * p, -1)
        return 0.5 * np.sum(p2 / m) / len(m)
    p0 = self.atoms.get_momenta()
    m = self._getmasses()
    p = np.array(p0, copy=1)
    dt = self.dt
    for i in range(2):
        self.q_past = self.q - dt * np.dot(p / m, self.inv_h)
        self._calculate_q_future(self.atoms.get_forces(md=True))
        p = np.dot(self.q_future - self.q_past, self.h / (2 * dt)) * m
        e = ekin(p)
        if e < 1e-05:
            return
        p = p0 - p + p0