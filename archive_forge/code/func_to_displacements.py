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
def to_displacements(self) -> None:
    """Converts positions of trajectory into displacements between consecutive frames.

        `base_positions` and `coords` should both be in fractional coords. Does
        not work for absolute coords because the atoms are to be wrapped into the
        simulation box.

        This is the opposite operation of `to_positions()`.
        """
    if self.coords_are_displacement:
        return
    displacements = np.subtract(self.coords, np.roll(self.coords, 1, axis=0))
    displacements[0] = np.zeros(np.shape(self.coords[0]))
    if self.lattice is not None:
        displacements = np.subtract(displacements, np.around(displacements))
    self.coords = displacements
    self.coords_are_displacement = True