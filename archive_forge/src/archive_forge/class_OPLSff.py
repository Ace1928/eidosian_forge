import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
class OPLSff:

    def __init__(self, fileobj=None, warnings=0):
        self.warnings = warnings
        self.data = {}
        if fileobj is not None:
            self.read(fileobj)

    def read(self, fileobj, comments='#'):

        def read_block(name, symlen, nvalues):
            """Read a data block.

            name: name of the block to store in self.data
            symlen: length of the symbol
            nvalues: number of values expected
            """
            if name not in self.data:
                self.data[name] = {}
            data = self.data[name]

            def add_line():
                line = fileobj.readline().strip()
                if not len(line):
                    return False
                line = line.split('#')[0]
                if len(line) > symlen:
                    symbol = line[:symlen]
                    words = line[symlen:].split()
                    if len(words) >= nvalues:
                        if nvalues == 1:
                            data[symbol] = float(words[0])
                        else:
                            data[symbol] = [float(word) for word in words[:nvalues]]
                return True
            while add_line():
                pass
        read_block('one', 2, 3)
        read_block('bonds', 5, 2)
        read_block('angles', 8, 2)
        read_block('dihedrals', 11, 4)
        read_block('cutoffs', 5, 1)
        self.bonds = BondData(self.data['bonds'])
        self.angles = AnglesData(self.data['angles'])
        self.dihedrals = DihedralsData(self.data['dihedrals'])
        self.cutoffs = CutoffList(self.data['cutoffs'])

    def write_lammps(self, atoms, prefix='lammps'):
        """Write input for a LAMMPS calculation."""
        self.prefix = prefix
        if hasattr(atoms, 'connectivities'):
            connectivities = atoms.connectivities
        else:
            btypes, blist = self.get_bonds(atoms)
            atypes, alist = self.get_angles()
            dtypes, dlist = self.get_dihedrals(alist, atypes)
            connectivities = {'bonds': blist, 'bond types': btypes, 'angles': alist, 'angle types': atypes, 'dihedrals': dlist, 'dihedral types': dtypes}
            self.write_lammps_definitions(atoms, btypes, atypes, dtypes)
            self.write_lammps_in()
        self.write_lammps_atoms(atoms, connectivities)

    def write_lammps_in(self):
        with open(self.prefix + '_in', 'w') as fileobj:
            self._write_lammps_in(fileobj)

    def _write_lammps_in(self, fileobj):
        fileobj.write('# LAMMPS relaxation (written by ASE)\n\nunits           metal\natom_style      full\nboundary        p p p\n#boundary       p p f\n\n')
        fileobj.write('read_data ' + self.prefix + '_atoms\n')
        fileobj.write('include  ' + self.prefix + '_opls\n')
        fileobj.write('\nkspace_style    pppm 1e-5\n#kspace_modify  slab 3.0\n\nneighbor        1.0 bin\nneigh_modify    delay 0 every 1 check yes\n\nthermo          1000\nthermo_style    custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms\n\ndump            1 all xyz 1000 dump_relax.xyz\ndump_modify     1 sort id\n\nrestart         100000 test_relax\n\nmin_style       fire\nminimize        1.0e-14 1.0e-5 100000 100000\n')

    def write_lammps_atoms(self, atoms, connectivities):
        """Write atoms input for LAMMPS"""
        with open(self.prefix + '_atoms', 'w') as fileobj:
            self._write_lammps_atoms(fileobj, atoms, connectivities)

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

    def update_neighbor_list(self, atoms):
        cut = 0.5 * max(self.data['cutoffs'].values())
        self.nl = NeighborList([cut] * len(atoms), skin=0, bothways=True, self_interaction=False)
        self.nl.update(atoms)
        self.atoms = atoms

    def get_bonds(self, atoms):
        """Find bonds and return them and their types"""
        cutoffs = CutoffList(self.data['cutoffs'])
        self.update_neighbor_list(atoms)
        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()
        bond_list = []
        bond_types = []
        for i, atom in enumerate(atoms):
            iname = types[tags[i]]
            indices, offsets = self.nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if j <= i:
                    continue
                jname = types[tags[j]]
                cut = cutoffs.value(iname, jname)
                if cut is None:
                    if self.warnings > 1:
                        print('Warning: cutoff %s-%s not found' % (iname, jname))
                    continue
                dist = np.linalg.norm(atom.position - atoms[j].position - np.dot(offset, cell))
                if dist > cut:
                    continue
                name, val = self.bonds.name_value(iname, jname)
                if name is None:
                    if self.warnings:
                        print('Warning: potential %s-%s not found' % (iname, jname))
                    continue
                if name not in bond_types:
                    bond_types.append(name)
                bond_list.append([bond_types.index(name), i, j])
        return (bond_types, bond_list)

    def get_angles(self, atoms=None):
        cutoffs = CutoffList(self.data['cutoffs'])
        if atoms is not None:
            self.update_neighbor_list(atoms)
        else:
            atoms = self.atoms
        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()
        ang_list = []
        ang_types = []
        for i, atom in enumerate(atoms):
            iname = types[tags[i]]
            indicesi, offsetsi = self.nl.get_neighbors(i)
            for j, offsetj in zip(indicesi, offsetsi):
                jname = types[tags[j]]
                cut = cutoffs.value(iname, jname)
                if cut is None:
                    continue
                dist = np.linalg.norm(atom.position - atoms[j].position - np.dot(offsetj, cell))
                if dist > cut:
                    continue
                for k, offsetk in zip(indicesi, offsetsi):
                    if k <= j:
                        continue
                    kname = types[tags[k]]
                    cut = cutoffs.value(iname, kname)
                    if cut is None:
                        continue
                    dist = np.linalg.norm(atom.position - np.dot(offsetk, cell) - atoms[k].position)
                    if dist > cut:
                        continue
                    name, val = self.angles.name_value(jname, iname, kname)
                    if name is None:
                        if self.warnings > 1:
                            print('Warning: angles %s-%s-%s not found' % (jname, iname, kname))
                        continue
                    if name not in ang_types:
                        ang_types.append(name)
                    ang_list.append([ang_types.index(name), j, i, k])
        return (ang_types, ang_list)

    def get_dihedrals(self, ang_types, ang_list):
        """Dihedrals derived from angles."""
        cutoffs = CutoffList(self.data['cutoffs'])
        atoms = self.atoms
        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()
        dih_list = []
        dih_types = []

        def append(name, i, j, k, L):
            if name not in dih_types:
                dih_types.append(name)
            index = dih_types.index(name)
            if [index, i, j, k, L] not in dih_list and [index, L, k, j, i] not in dih_list:
                dih_list.append([index, i, j, k, L])
        for angle in ang_types:
            L, i, j, k = angle
            iname = types[tags[i]]
            jname = types[tags[j]]
            kname = types[tags[k]]
            indicesi, offsetsi = self.nl.get_neighbors(i)
            for L, offsetl in zip(indicesi, offsetsi):
                if L == j:
                    continue
                lname = types[tags[L]]
                cut = cutoffs.value(iname, lname)
                if cut is None:
                    continue
                dist = np.linalg.norm(atoms[i].position - atoms[L].position - np.dot(offsetl, cell))
                if dist > cut:
                    continue
                name, val = self.dihedrals.name_value(lname, iname, jname, kname)
                if name is None:
                    continue
                append(name, L, i, j, k)
            indicesk, offsetsk = self.nl.get_neighbors(k)
            for L, offsetl in zip(indicesk, offsetsk):
                if L == j:
                    continue
                lname = types[tags[L]]
                cut = cutoffs.value(kname, lname)
                if cut is None:
                    continue
                dist = np.linalg.norm(atoms[k].position - atoms[L].position - np.dot(offsetl, cell))
                if dist > cut:
                    continue
                name, val = self.dihedrals.name_value(iname, jname, kname, lname)
                if name is None:
                    continue
                append(name, i, j, k, L)
        return (dih_types, dih_list)

    def write_lammps_definitions(self, atoms, btypes, atypes, dtypes):
        """Write force field definitions for LAMMPS."""
        with open(self.prefix + '_opls', 'w') as fd:
            self._write_lammps_definitions(fd, atoms, btypes, atypes, dtypes)

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