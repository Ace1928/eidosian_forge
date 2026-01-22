from __future__ import annotations
import itertools
import warnings
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Composition, DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
def _get_site_props(self, frames: int | list[int]) -> SitePropsType | None:
    """Slice site properties."""
    if self.site_properties is None:
        return None
    if isinstance(self.site_properties, dict):
        return self.site_properties
    if isinstance(self.site_properties, list):
        if isinstance(frames, int):
            return self.site_properties[frames]
        if isinstance(frames, list):
            return [self.site_properties[i] for i in frames]
        raise ValueError('Unexpected frames type.')
    raise ValueError('Unexpected site_properties type.')