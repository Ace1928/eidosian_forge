import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
def _get_frame_positions(f):
    """Get positions of frames in HISTORY file."""
    init_pos = f.tell()
    f.seek(0)
    rl = len(f.readline())
    items = f.readline().strip().split()
    if len(items) == 5:
        classic = False
    elif len(items) == 3:
        classic = True
    else:
        raise RuntimeError('Cannot determine version of HISTORY file format.')
    levcfg, imcon, natoms = [int(x) for x in items[0:3]]
    if classic:
        startpos = f.tell()
        pos = []
        line = True
        while line:
            line = f.readline()
            if 'timestep' in line:
                pos.append(f.tell())
        f.seek(startpos)
    else:
        nframes = int(items[3])
        pos = [((natoms * (levcfg + 2) + 4) * i + 3) * rl for i in range(nframes)]
    f.seek(init_pos)
    return (levcfg, imcon, natoms, pos)