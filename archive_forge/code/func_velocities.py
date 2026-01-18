from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def velocities(self) -> list[Vector3D]:
    """The velocities of the atoms"""
    if 'velocities' not in self._cache:
        self._cache['species'], self._cache['coords'], self._cache['velocities'], self._cache['lattice'] = self._parse_lattice_atom_pos()
    return self._cache['velocities']