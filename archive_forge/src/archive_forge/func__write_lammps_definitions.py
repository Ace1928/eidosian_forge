import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def _write_lammps_definitions(self, fileobj, atoms, btypes, atypes, dtypes):
    fileobj.write('# OPLS potential\n')
    fileobj.write('# write_lammps' + str(time.asctime(time.localtime(time.time()))))
    if len(btypes):
        fileobj.write('\n# bonds\n')
        fileobj.write('bond_style      harmonic\n')
        for ib, btype in enumerate(btypes):
            fileobj.write('bond_coeff %6d' % (ib + 1))
            for value in self.bonds.nvh[btype]:
                fileobj.write(' ' + str(value))
            fileobj.write(' # ' + btype + '\n')
    if len(atypes):
        fileobj.write('\n# angles\n')
        fileobj.write('angle_style      harmonic\n')
        for ia, atype in enumerate(atypes):
            fileobj.write('angle_coeff %6d' % (ia + 1))
            for value in self.angles.nvh[atype]:
                fileobj.write(' ' + str(value))
            fileobj.write(' # ' + atype + '\n')
    if len(dtypes):
        fileobj.write('\n# dihedrals\n')
        fileobj.write('dihedral_style      opls\n')
        for i, dtype in enumerate(dtypes):
            fileobj.write('dihedral_coeff %6d' % (i + 1))
            for value in self.dihedrals.nvh[dtype]:
                fileobj.write(' ' + str(value))
            fileobj.write(' # ' + dtype + '\n')
    fileobj.write('\n# L-J parameters\n')
    fileobj.write('pair_style lj/cut/coul/long 10.0 7.4' + ' # consider changing these parameters\n')
    fileobj.write('special_bonds lj/coul 0.0 0.0 0.5\n')
    data = self.data['one']
    for ia, atype in enumerate(atoms.types):
        if len(atype) < 2:
            atype = atype + ' '
        fileobj.write('pair_coeff ' + str(ia + 1) + ' ' + str(ia + 1))
        for value in data[atype][:2]:
            fileobj.write(' ' + str(value))
        fileobj.write(' # ' + atype + '\n')
    fileobj.write('pair_modify shift yes mix geometric\n')
    fileobj.write('\n# charges\n')
    for ia, atype in enumerate(atoms.types):
        if len(atype) < 2:
            atype = atype + ' '
        fileobj.write('set type ' + str(ia + 1))
        fileobj.write(' charge ' + str(data[atype][2]))
        fileobj.write(' # ' + atype + '\n')