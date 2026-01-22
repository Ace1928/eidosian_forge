from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def _get_site_composition(self) -> None:
    """
        Get Icet-format composition from structure.

        Returns:
            Dict with sublattice compositions specified by uppercase letters,
                e.g., In_x Ga_1-x As becomes:
                {
                    "A": {"In": x, "Ga": 1 - x},
                    "B": {"As": 1}
                }
        """
    uppercase_letters = list(ascii_uppercase)
    idx = 0
    self.composition: dict[str, dict] = {}
    for idx, site in enumerate(self._structure):
        site_comp = site.species.as_dict()
        if site_comp not in self.composition.values():
            self.composition[uppercase_letters[idx]] = site_comp
            idx += 1