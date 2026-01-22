from math import pi
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha
class TestPotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        E = 0.0
        R = atoms.positions
        F = np.zeros_like(R)
        for a, r in enumerate(R):
            D = R - r
            d = (D ** 2).sum(1) ** 0.5
            x = d - 1.0
            E += np.vdot(x, x)
            d[a] = 1
            F -= (x / d)[:, None] * D
        energy = 0.25 * E
        self.results = {'energy': energy, 'forces': F}