import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
Does the actual mutation.