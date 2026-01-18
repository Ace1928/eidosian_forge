import os
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLength
from ase.db import connect
from ase.io import read
from ase.build import molecule
Make sure user=someone works.  Is called username in SQLite.