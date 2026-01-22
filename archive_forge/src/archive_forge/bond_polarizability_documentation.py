from typing import Tuple
import numpy as np
from ase.units import Bohr, Ha
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
Sum up the bond polarizability from all bonds

        Parameters
        ----------
        atoms: Atoms object
        radiicut: float
          Bonds are counted up to
          radiicut * (sum of covalent radii of the pairs)
          Default: 1.5

        Returns
        -------
        polarizability tensor with unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        