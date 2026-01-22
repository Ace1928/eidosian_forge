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
@staticmethod
def _combine_lattice(lat1: np.ndarray, lat2: np.ndarray, len1: int, len2: int) -> tuple[np.ndarray, bool]:
    """Helper function to combine trajectory lattice."""
    if lat1.ndim == lat2.ndim == 2:
        constant_lat = True
        lat = lat1
    else:
        constant_lat = False
        if lat1.ndim == 2:
            lat1 = np.tile(lat1, (len1, 1, 1))
        if lat2.ndim == 2:
            lat2 = np.tile(lat2, (len2, 1, 1))
        lat = np.concatenate((lat1, lat2))
    return (lat, constant_lat)