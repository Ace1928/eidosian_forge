from math import pi
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha
class FreeElectrons(Calculator):
    """Free-electron band calculator.

    Parameters:

    nvalence: int
        Number of electrons
    kpts: dict
        K-point specification.

    Example:

    >>> calc = FreeElectrons(nvalence=1, kpts={'path': 'GXL'})
    """
    implemented_properties = ['energy']
    default_parameters = {'kpts': np.zeros((1, 3)), 'nvalence': 0.0, 'nbands': 20, 'gridsize': 7}

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms)
        self.kpts = kpts2ndarray(self.parameters.kpts, atoms)
        icell = atoms.cell.reciprocal() * 2 * np.pi * Bohr
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
        eps = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
        eps.sort()
        self.eigenvalues = eps[:, :self.parameters.nbands] * Ha
        self.results = {'energy': 0.0}

    def get_eigenvalues(self, kpt, spin=0):
        assert spin == 0
        return self.eigenvalues[kpt].copy()

    def get_fermi_level(self):
        v = self.atoms.get_volume() / Bohr ** 3
        kF = (self.parameters.nvalence / v * 3 * np.pi ** 2) ** (1 / 3)
        return 0.5 * kF ** 2 * Ha

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_number_of_spins(self):
        return 1