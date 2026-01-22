import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def _write_lammps_in(self, fileobj):
    fileobj.write('# LAMMPS relaxation (written by ASE)\n\nunits           metal\natom_style      full\nboundary        p p p\n#boundary       p p f\n\n')
    fileobj.write('read_data ' + self.prefix + '_atoms\n')
    fileobj.write('include  ' + self.prefix + '_opls\n')
    fileobj.write('\nkspace_style    pppm 1e-5\n#kspace_modify  slab 3.0\n\nneighbor        1.0 bin\nneigh_modify    delay 0 every 1 check yes\n\nthermo          1000\nthermo_style    custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms\n\ndump            1 all xyz 1000 dump_relax.xyz\ndump_modify     1 sort id\n\nrestart         100000 test_relax\n\nmin_style       fire\nminimize        1.0e-14 1.0e-5 100000 100000\n')