import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _initialize_eta_h(self):
    self.h_past = self.h - self.dt * np.dot(self.h, self.eta)
    if self.pfactor_given is None:
        deltaeta = np.zeros(6, float)
    else:
        deltaeta = -self.dt * self.pfact * linalg.det(self.h) * (self.stresscalculator() - self.externalstress)
    if self.frac_traceless == 1:
        self.eta_past = self.eta - self.mask * self._makeuppertriangular(deltaeta)
    else:
        trace_part, traceless_part = self._separatetrace(self._makeuppertriangular(deltaeta))
        self.eta_past = self.eta - trace_part - self.frac_traceless * traceless_part