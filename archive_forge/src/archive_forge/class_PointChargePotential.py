import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
class PointChargePotential:

    def __init__(self, mmcharges):
        """Point-charge potential for TIP3P.

        Only used for testing QMMM.
        """
        self.mmcharges = mmcharges
        self.mmpositions = None
        self.mmforces = None

    def set_positions(self, mmpositions, com_pv=None):
        self.mmpositions = mmpositions

    def calculate(self, qmcharges, qmpositions):
        energy = 0.0
        self.mmforces = np.zeros_like(self.mmpositions)
        qmforces = np.zeros_like(qmpositions)
        for C, R, F in zip(self.mmcharges, self.mmpositions, self.mmforces):
            d = qmpositions - R
            r2 = (d ** 2).sum(1)
            e = units.Hartree * units.Bohr * C * r2 ** (-0.5) * qmcharges
            energy += e.sum()
            f = (e / r2)[:, np.newaxis] * d
            qmforces += f
            F -= f.sum(0)
        self.mmpositions = None
        return (energy, qmforces)

    def get_forces(self, calc):
        return self.mmforces