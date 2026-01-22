import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
class AnglesData:

    def __init__(self, name_value_hash):
        self.nvh = name_value_hash

    def name_value(self, aname, bname, cname):
        for name in [twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname), twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)]:
            if name in self.nvh:
                return (name, self.nvh[name])
        return (None, None)