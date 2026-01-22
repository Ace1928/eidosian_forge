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
class Cp2kValidationError(Exception):
    """
    Cp2k Validation Exception. Not exhausted. May raise validation
    errors for features which actually do work if using a newer version
    of cp2k.
    """
    CP2K_VERSION = 'v2022.1'

    def __init__(self, message) -> None:
        message = f'CP2K {self.CP2K_VERSION}: {message}'
        super().__init__(message)