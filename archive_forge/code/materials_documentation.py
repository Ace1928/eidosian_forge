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
Create TransformedStructure from SNL.

        Args:
            snl (StructureNL): Starting snl

        Returns:
            TransformedStructure
        