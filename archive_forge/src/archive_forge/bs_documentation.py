from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath
Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure or Molecule
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns:
            dict: The updated for the parameters for the output section of FHI-aims
        