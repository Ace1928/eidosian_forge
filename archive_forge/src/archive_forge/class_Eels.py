from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
class Eels(MSONable):
    """Parse'eels.dat' file."""

    def __init__(self, data):
        """
        Args:
            data (): Eels data.
        """
        self.data = np.array(data)

    @property
    def energies(self):
        """Returns the energies in eV."""
        return self.data[:, 0]

    @property
    def total_spectrum(self):
        """Returns the total eels spectrum."""
        return self.data[:, 1]

    @property
    def atomic_background(self):
        """Returns: atomic background."""
        return self.data[:, 2]

    @property
    def fine_structure(self):
        """Returns: Fine structure of EELS."""
        return self.data[:, 3]

    @classmethod
    def from_file(cls, eels_dat_file: str='eels.dat') -> Self:
        """
        Parse eels spectrum.

        Args:
            eels_dat_file (str): filename and path for eels.dat

        Returns:
            Eels
        """
        data = np.loadtxt(eels_dat_file)
        return cls(data)

    def as_dict(self) -> dict:
        """Returns dict representations of Xmu object."""
        dct = MSONable.as_dict(self)
        dct['data'] = self.data.tolist()
        return dct