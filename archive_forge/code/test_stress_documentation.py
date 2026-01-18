import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
two atoms at potential minimum