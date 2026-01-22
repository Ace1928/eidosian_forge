from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def all_forces(self) -> list[list[Vector3D]]:
    """The forces for all images in the calculation."""
    all_forces_array = [res.site_properties.get('force', None) for res in self._results]
    return [af.tolist() if isinstance(af, np.ndarray) else af for af in all_forces_array]