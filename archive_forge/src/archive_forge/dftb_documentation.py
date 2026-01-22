import os
import numpy as np
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
from ase.units import Hartree, Bohr
Read Forces from dftb output file (results.tag).