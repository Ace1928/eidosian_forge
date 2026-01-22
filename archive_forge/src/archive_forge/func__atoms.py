import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.spectrum.band_structure import calculate_band_structure
from ase.calculators.test import FreeElectrons
from ase.cell import Cell
def _atoms(cell):
    atoms = Atoms(cell=cell, pbc=True)
    atoms.calc = FreeElectrons()
    return atoms