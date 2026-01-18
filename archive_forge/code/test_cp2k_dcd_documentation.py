import subprocess
import numpy as np
import pytest
from ase.build import molecule
from ase import io
from ase.io.cp2k import iread_cp2k_dcd
from ase.calculators.calculator import compare_atoms
Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
