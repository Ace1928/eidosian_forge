import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def _write_lammps_atoms(self, fileobj, atoms, connectivities):
    fileobj.write(fileobj.name + ' (by ' + str(self.__class__) + ')\n\n')
    fileobj.write(str(len(atoms)) + ' atoms\n')
    fileobj.write(str(len(atoms.types)) + ' atom types\n')
    blist = connectivities['bonds']
    if len(blist):
        btypes = connectivities['bond types']
        fileobj.write(str(len(blist)) + ' bonds\n')
        fileobj.write(str(len(btypes)) + ' bond types\n')
    alist = connectivities['angles']
    if len(alist):
        atypes = connectivities['angle types']
        fileobj.write(str(len(alist)) + ' angles\n')
        fileobj.write(str(len(atypes)) + ' angle types\n')
    dlist = connectivities['dihedrals']
    if len(dlist):
        dtypes = connectivities['dihedral types']
        fileobj.write(str(len(dlist)) + ' dihedrals\n')
        fileobj.write(str(len(dtypes)) + ' dihedral types\n')
    p = Prism(atoms.get_cell())
    xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism()
    fileobj.write('\n0.0 %s  xlo xhi\n' % xhi)
    fileobj.write('0.0 %s  ylo yhi\n' % yhi)
    fileobj.write('0.0 %s  zlo zhi\n' % zhi)
    if p.is_skewed():
        fileobj.write(f'{xy} {xz} {yz}  xy xz yz\n')
    fileobj.write('\nAtoms\n\n')
    tag = atoms.get_tags()
    if atoms.has('molid'):
        molid = atoms.get_array('molid')
    else:
        molid = [1] * len(atoms)
    for i, r in enumerate(p.vector_to_lammps(atoms.get_positions())):
        atype = atoms.types[tag[i]]
        if len(atype) < 2:
            atype = atype + ' '
        q = self.data['one'][atype][2]
        fileobj.write('%6d %3d %3d %s %s %s %s' % ((i + 1, molid[i], tag[i] + 1, q) + tuple(r)))
        fileobj.write(' # ' + atoms.types[tag[i]] + '\n')
    velocities = atoms.get_velocities()
    if velocities is not None:
        velocities = p.vector_to_lammps(atoms.get_velocities())
        fileobj.write('\nVelocities\n\n')
        for i, v in enumerate(velocities):
            fileobj.write('%6d %g %g %g\n' % (i + 1, v[0], v[1], v[2]))
    fileobj.write('\nMasses\n\n')
    for i, typ in enumerate(atoms.types):
        cs = atoms.split_symbol(typ)[0]
        fileobj.write('%6d %g # %s -> %s\n' % (i + 1, atomic_masses[chemical_symbols.index(cs)], typ, cs))
    if blist:
        fileobj.write('\nBonds\n\n')
        for ib, bvals in enumerate(blist):
            fileobj.write('%8d %6d %6d %6d ' % (ib + 1, bvals[0] + 1, bvals[1] + 1, bvals[2] + 1))
            if bvals[0] in btypes:
                fileobj.write('# ' + btypes[bvals[0]])
            fileobj.write('\n')
    if alist:
        fileobj.write('\nAngles\n\n')
        for ia, avals in enumerate(alist):
            fileobj.write('%8d %6d %6d %6d %6d ' % (ia + 1, avals[0] + 1, avals[1] + 1, avals[2] + 1, avals[3] + 1))
            if avals[0] in atypes:
                fileobj.write('# ' + atypes[avals[0]])
            fileobj.write('\n')
    if dlist:
        fileobj.write('\nDihedrals\n\n')
        for i, dvals in enumerate(dlist):
            fileobj.write('%8d %6d %6d %6d %6d %6d ' % (i + 1, dvals[0] + 1, dvals[1] + 1, dvals[2] + 1, dvals[3] + 1, dvals[4] + 1))
            if dvals[0] in dtypes:
                fileobj.write('# ' + dtypes[dvals[0]])
            fileobj.write('\n')