import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def _set_efermi(self, efermi):
    """Set the Fermi level."""
    ef = efermi - self._efermi
    self._efermi = efermi
    self._total_dos[0, :] = self._total_dos[0, :] - ef
    try:
        self._site_dos[:, 0, :] = self._site_dos[:, 0, :] - ef
    except IndexError:
        pass