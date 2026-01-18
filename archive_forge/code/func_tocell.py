from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def tocell(self) -> Cell:
    """Return this lattice as a :class:`~ase.cell.Cell` object."""
    cell = self._cell(**self._parameters)
    return Cell(cell)