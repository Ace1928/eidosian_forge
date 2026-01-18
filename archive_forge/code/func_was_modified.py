from __future__ import annotations
import datetime
import json
import re
from typing import TYPE_CHECKING, Any
from warnings import warn
from monty.json import MSONable, jsanitize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.util.provenance import StructureNL
@property
def was_modified(self) -> bool:
    """Boolean describing whether the last transformation on the structure
        made any alterations to it one example of when this would return false
        is in the case of performing a substitution transformation on the
        structure when the specie to replace isn't in the structure.
        """
    return self.final_structure != self.structures[-2]