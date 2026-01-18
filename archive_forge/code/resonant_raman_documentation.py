import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
Overlap is determined as

            ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
            