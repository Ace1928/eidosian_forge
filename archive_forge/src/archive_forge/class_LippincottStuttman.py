from typing import Tuple
import numpy as np
from ase.units import Bohr, Ha
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
class LippincottStuttman:
    atomic_polarizability = {'B': 1.358, 'C': 0.978, 'N': 0.743, 'O': 0.592, 'Al': 3.918, 'Si': 2.988}
    reduced_eletronegativity = {'B': 0.538, 'C': 0.846, 'N': 0.927, 'O': 1.0, 'Al': 0.533, 'Si': 0.583}

    def __call__(self, el1: str, el2: str, length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        alpha1 = self.atomic_polarizability[el1]
        alpha2 = self.atomic_polarizability[el2]
        ren1 = self.reduced_eletronegativity[el1]
        ren2 = self.reduced_eletronegativity[el2]
        sigma = 1.0
        if el1 != el2:
            sigma = np.exp(-(ren1 - ren2) ** 2 / 4)
        alphal = sigma * length ** 4 / (4 ** 4 * alpha1 * alpha2) ** (1.0 / 6)
        alphap = (ren1 ** 2 * alpha1 + ren2 ** 2 * alpha2) / (ren1 ** 2 + ren2 ** 2)
        return (alphal, alphap)