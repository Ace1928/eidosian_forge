import os
from warnings import warn
from glob import glob
import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.utils import writer
Writes structure to EON reactant.con file
    Multiple snapshots are allowed.