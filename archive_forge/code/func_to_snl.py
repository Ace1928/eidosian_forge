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
def to_snl(self, authors: list[str], **kwargs) -> StructureNL:
    """
        Generate a StructureNL from TransformedStructure.

        Args:
            authors (List[str]): List of authors contributing to the generated StructureNL.
            **kwargs (Any): All kwargs supported by StructureNL.

        Returns:
            StructureNL: The generated StructureNL object.
        """
    if self.other_parameters:
        warn('Data in TransformedStructure.other_parameters discarded during type conversion to SNL')
    history = []
    for hist in self.history:
        snl_metadata = hist.pop('_snl', {})
        history.append({'name': snl_metadata.pop('name', 'pymatgen'), 'url': snl_metadata.pop('url', 'http://pypi.python.org/pypi/pymatgen'), 'description': hist})
    return StructureNL(self.final_structure, authors, history=history, **kwargs)