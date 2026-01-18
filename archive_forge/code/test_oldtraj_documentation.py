from base64 import b64encode, b64decode
from pathlib import Path
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.trajectory import Trajectory
Run this with an old version of ASE.

    Did it with 3.18.1.
    