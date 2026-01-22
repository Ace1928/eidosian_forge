from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def activate_epr(self, **kwargs) -> None:
    """Calculate g-tensor. Requires localize. Suggested with GAPW."""
    if not self.check('force_eval/properties/linres/localize'):
        self.activate_localize()
    self['FORCE_EVAL']['PROPERTIES']['LINRES'].insert(Section('EPR', **kwargs))
    self['FORCE_EVAL']['PROPERTIES']['LINRES']['EPR'].update({'PRINT': {'G_TENSOR': {}}})