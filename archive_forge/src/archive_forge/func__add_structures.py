from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
def _add_structures(ordered_structures, ordered_structures_origins, structures_to_add, origin=''):
    """Transformations with return_ranked_list can return either
            just Structures or dicts (or sometimes lists!) -- until this
            is fixed, we use this function to concat structures given
            by the transformation.
            """
    if structures_to_add:
        if isinstance(structures_to_add, Structure):
            structures_to_add = [structures_to_add]
        structures_to_add = [s['structure'] if isinstance(s, dict) else s for s in structures_to_add]
        ordered_structures += structures_to_add
        ordered_structures_origins += [origin] * len(structures_to_add)
        self.logger.info(f'Adding {len(structures_to_add)} ordered structures: {origin}')
    return (ordered_structures, ordered_structures_origins)