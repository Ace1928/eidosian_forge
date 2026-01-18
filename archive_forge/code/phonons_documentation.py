from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
Write modes to trajectory file.

        Parameters:

        q_c: ndarray
            q-vector of the modes.
        branches: int or list
            Branch index of modes.
        kT: float
            Temperature in units of eV. Determines the amplitude of the atomic
            displacements in the modes.
        born: bool
            Include non-analytic contribution to the force constants at q -> 0.
        repeat: tuple
            Repeat atoms (l, m, n) times in the directions of the lattice
            vectors. Displacements of atoms in repeated cells carry a Bloch
            phase factor given by the q-vector and the cell lattice vector R_m.
        nimages: int
            Number of images in an oscillation.
        center: bool
            Center atoms in unit cell if True (default: False).

        