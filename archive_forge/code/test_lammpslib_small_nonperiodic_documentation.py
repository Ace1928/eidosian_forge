import pytest
import numpy as np
from ase import Atoms
Test that lammpslib handle nonperiodic cases where the cell size
    in some directions is small (for example for a dimer)