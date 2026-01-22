from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
Creating bulk systems.

    Crystal structure and lattice constant(s) will be guessed if not
    provided.

    name: str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure: str
        Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, 
        orthorhombic, mlc, diamond, zincblende, rocksalt, cesiumchloride, 
        fluorite or wurtzite.
    a: float
        Lattice constant.
    b: float
        Lattice constant.  If only a and b is given, b will be interpreted
        as c instead.
    c: float
        Lattice constant.
    alpha: float
        Angle in degrees for rhombohedral lattice.
    covera: float
        c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
    u: float
        Internal coordinate for Wurtzite structure.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    cubic: bool
        Construct cubic unit cell if possible.
    