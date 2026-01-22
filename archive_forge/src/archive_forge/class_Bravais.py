import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
class Bravais:
    """Bravais lattice factory.

    This is a base class for the objects producing various lattices
    (SC, FCC, ...).
    """
    other = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    bravais_basis: Optional[Sequence[Sequence[float]]] = None
    element_basis: Optional[Sequence[int]] = None
    chop_tolerance = 1e-10

    def __call__(self, symbol, directions=(None, None, None), miller=(None, None, None), size=(1, 1, 1), latticeconstant=None, pbc=True, align=True, debug=0):
        """Create a lattice."""
        self.size = size
        self.pbc = pbc
        self.debug = debug
        self.process_element(symbol)
        self.find_directions(directions, miller)
        if self.debug:
            self.print_directions_and_miller()
        self.convert_to_natural_basis()
        if self.debug >= 2:
            self.print_directions_and_miller(' (natural basis)')
        if latticeconstant is None:
            if self.element_basis is None:
                self.latticeconstant = self.get_lattice_constant()
            else:
                raise ValueError('A lattice constant must be specified for a compound')
        else:
            self.latticeconstant = latticeconstant
        if self.debug:
            print('Expected number of atoms in unit cell:', self.calc_num_atoms())
        if self.debug >= 2:
            print('Bravais lattice basis:', self.bravais_basis)
            if self.bravais_basis is not None:
                print(' ... in natural basis:', self.natural_bravais_basis)
        self.make_crystal_basis()
        self.make_unit_cell()
        if align:
            self.align()
        return self.make_list_of_atoms()

    def align(self):
        """Align the first axis along x-axis and the second in the x-y plane."""
        degree = 180 / np.pi
        if self.debug >= 2:
            print('Basis before alignment:')
            print(self.basis)
        if self.basis[0][0] ** 2 + self.basis[0][2] ** 2 < 0.01 * self.basis[0][1] ** 2:
            t = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
            self.basis = np.dot(self.basis, t)
            transf = t
            if self.debug >= 2:
                print('Rotating -90 degrees around z axis for numerical stability.')
                print(self.basis)
        else:
            transf = np.identity(3, float)
        assert abs(np.linalg.det(transf) - 1) < 1e-06
        theta = math.atan2(self.basis[0, 2], self.basis[0, 0])
        t = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        self.basis = np.dot(self.basis, t)
        transf = np.dot(transf, t)
        if self.debug >= 2:
            print('Rotating %f degrees around y axis.' % (-theta * degree,))
            print(self.basis)
        assert abs(np.linalg.det(transf) - 1) < 1e-06
        theta = math.atan2(self.basis[0, 1], self.basis[0, 0])
        t = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        self.basis = np.dot(self.basis, t)
        transf = np.dot(transf, t)
        if self.debug >= 2:
            print('Rotating %f degrees around z axis.' % (-theta * degree,))
            print(self.basis)
        assert abs(np.linalg.det(transf) - 1) < 1e-06
        theta = math.atan2(self.basis[1, 2], self.basis[1, 1])
        t = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        self.basis = np.dot(self.basis, t)
        transf = np.dot(transf, t)
        if self.debug >= 2:
            print('Rotating %f degrees around x axis.' % (-theta * degree,))
            print(self.basis)
        assert abs(np.linalg.det(transf) - 1) < 1e-06
        self.atoms = np.dot(self.atoms, transf)
        self.miller_basis = np.dot(self.miller_basis, transf)

    def make_list_of_atoms(self):
        """Repeat the unit cell."""
        nrep = self.size[0] * self.size[1] * self.size[2]
        if nrep <= 0:
            raise ValueError('Cannot create a non-positive number of unit cells')
        a2 = []
        e2 = []
        for i in range(self.size[0]):
            offset = self.basis[0] * i
            a2.append(self.atoms + offset[np.newaxis, :])
            e2.append(self.elements)
        atoms = np.concatenate(a2)
        elements = np.concatenate(e2)
        a2 = []
        e2 = []
        for j in range(self.size[1]):
            offset = self.basis[1] * j
            a2.append(atoms + offset[np.newaxis, :])
            e2.append(elements)
        atoms = np.concatenate(a2)
        elements = np.concatenate(e2)
        a2 = []
        e2 = []
        for k in range(self.size[2]):
            offset = self.basis[2] * k
            a2.append(atoms + offset[np.newaxis, :])
            e2.append(elements)
        atoms = np.concatenate(a2)
        elements = np.concatenate(e2)
        del a2, e2
        assert len(atoms) == nrep * len(self.atoms)
        basis = np.array([[self.size[0], 0, 0], [0, self.size[1], 0], [0, 0, self.size[2]]])
        basis = np.dot(basis, self.basis)
        basis = np.where(np.abs(basis) < self.chop_tolerance, 0.0, basis)
        lattice = Lattice(positions=atoms, cell=basis, numbers=elements, pbc=self.pbc)
        lattice.millerbasis = self.miller_basis
        lattice._addsorbate_info_size = np.array(self.size[:2])
        return lattice

    def process_element(self, element):
        """Extract atomic number from element"""
        if self.element_basis is None:
            if isinstance(element, str):
                self.atomicnumber = ase.data.atomic_numbers[element]
            elif isinstance(element, int):
                self.atomicnumber = element
            else:
                raise TypeError('The symbol argument must be a string or an atomic number.')
        else:
            atomicnumber = []
            try:
                if len(element) != max(self.element_basis) + 1:
                    oops = True
                else:
                    oops = False
            except TypeError:
                oops = True
            if oops:
                raise TypeError(('The symbol argument must be a sequence of length %d' + ' (one for each kind of lattice position') % (max(self.element_basis) + 1,))
            for e in element:
                if isinstance(e, str):
                    atomicnumber.append(ase.data.atomic_numbers[e])
                elif isinstance(e, int):
                    atomicnumber.append(e)
                else:
                    raise TypeError('The symbols argument must be a sequence of strings or atomic numbers.')
            self.atomicnumber = [atomicnumber[i] for i in self.element_basis]
            assert len(self.atomicnumber) == len(self.bravais_basis)

    def convert_to_natural_basis(self):
        """Convert directions and miller indices to the natural basis."""
        self.directions = np.dot(self.directions, self.inverse_basis)
        if self.bravais_basis is not None:
            self.natural_bravais_basis = np.dot(self.bravais_basis, self.inverse_basis)
        for i in (0, 1, 2):
            self.directions[i] = reduceindex(self.directions[i])
        for i in (0, 1, 2):
            j, k = self.other[i]
            self.miller[i] = reduceindex(self.handedness * cross(self.directions[j], self.directions[k]))

    def calc_num_atoms(self):
        v = int(round(abs(np.linalg.det(self.directions))))
        if self.bravais_basis is None:
            return v
        else:
            return v * len(self.bravais_basis)

    def make_unit_cell(self):
        """Make the unit cell."""
        self.natoms = self.calc_num_atoms()
        self.nput = 0
        self.atoms = np.zeros((self.natoms, 3), float)
        self.elements = np.zeros(self.natoms, int)
        self.farpoint = sum(self.directions)
        sqrad = 0
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    vect = i * self.directions[0] + j * self.directions[1] + k * self.directions[2]
                    if np.dot(vect, vect) > sqrad:
                        sqrad = np.dot(vect, vect)
        del i, j, k
        for istart, istep in ((0, 1), (-1, -1)):
            i = istart
            icont = True
            while icont:
                nj = 0
                for jstart, jstep in ((0, 1), (-1, -1)):
                    j = jstart
                    jcont = True
                    while jcont:
                        nk = 0
                        for kstart, kstep in ((0, 1), (-1, -1)):
                            k = kstart
                            kcont = True
                            while kcont:
                                point = np.array((i, j, k))
                                if self.inside(point):
                                    self.put_atom(point)
                                    nk += 1
                                    nj += 1
                                if np.dot(point, point) > sqrad:
                                    assert not self.inside(point)
                                    kcont = False
                                k += kstep
                        if i * i + j * j > sqrad:
                            jcont = False
                        j += jstep
                if i * i > sqrad:
                    icont = False
                i += istep
        assert self.nput == self.natoms

    def inside(self, point):
        """Is a point inside the unit cell?"""
        return np.dot(self.miller[0], point) >= 0 and np.dot(self.miller[0], point - self.farpoint) < 0 and (np.dot(self.miller[1], point) >= 0) and (np.dot(self.miller[1], point - self.farpoint) < 0) and (np.dot(self.miller[2], point) >= 0) and (np.dot(self.miller[2], point - self.farpoint) < 0)

    def put_atom(self, point):
        """Place an atom given its integer coordinates."""
        if self.bravais_basis is None:
            pos = np.dot(point, self.crystal_basis)
            if self.debug >= 2:
                print('Placing an atom at (%d,%d,%d) ~ (%.3f, %.3f, %.3f).' % (tuple(point) + tuple(pos)))
            self.atoms[self.nput] = pos
            self.elements[self.nput] = self.atomicnumber
            self.nput += 1
        else:
            for i, offset in enumerate(self.natural_bravais_basis):
                pos = np.dot(point + offset, self.crystal_basis)
                if self.debug >= 2:
                    print('Placing an atom at (%d+%f, %d+%f, %d+%f) ~ (%.3f, %.3f, %.3f).' % (point[0], offset[0], point[1], offset[1], point[2], offset[2], pos[0], pos[1], pos[2]))
                self.atoms[self.nput] = pos
                if self.element_basis is None:
                    self.elements[self.nput] = self.atomicnumber
                else:
                    self.elements[self.nput] = self.atomicnumber[i]
                self.nput += 1

    def find_directions(self, directions, miller):
        """
        Find missing directions and miller indices from the specified ones.
        """
        directions = np.asarray(directions).tolist()
        miller = list(miller)
        if directions == [None, None, None] and miller == [None, None, None]:
            directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        change = 1
        while change:
            change = False
            missing = 0
            for i in (0, 1, 2):
                j, k = self.other[i]
                if directions[i] is None:
                    missing += 1
                    if miller[j] is not None and miller[k] is not None:
                        directions[i] = reduceindex(cross(miller[j], miller[k]))
                        change = True
                        if self.debug >= 2:
                            print('Calculating directions[%d] from miller indices' % i)
                if miller[i] is None:
                    missing += 1
                    if directions[j] is not None and directions[k] is not None:
                        miller[i] = reduceindex(cross(directions[j], directions[k]))
                        change = True
                        if self.debug >= 2:
                            print('Calculating miller[%d] from directions' % i)
        if missing:
            raise ValueError('Specification of directions and miller indices is incomplete.')
        self.directions = np.array(directions)
        self.miller = np.array(miller)
        if abs(np.linalg.det(self.directions)) < 1e-10:
            raise ValueError('The direction vectors are linearly dependent (unit cell volume would be zero)')
        if np.linalg.det(self.directions) < 0:
            print('WARNING: Creating a left-handed coordinate system!')
            self.miller = -self.miller
            self.handedness = -1
        else:
            self.handedness = 1
        for i in (0, 1, 2):
            j, k = self.other[i]
            m = reduceindex(self.handedness * cross(self.directions[j], self.directions[k]))
            if sum(np.not_equal(m, self.miller[i])):
                print('ERROR: Miller index %s is inconsisten with directions %d and %d' % (i, j, k))
                print('Miller indices:')
                print(str(self.miller))
                print('Directions:')
                print(str(self.directions))
                raise ValueError('Inconsistent specification of miller indices and directions.')

    def print_directions_and_miller(self, txt=''):
        """Print direction vectors and Miller indices."""
        print('Direction vectors of unit cell%s:' % (txt,))
        for i in (0, 1, 2):
            print('   ', self.directions[i])
        print('Miller indices of surfaces%s:' % (txt,))
        for i in (0, 1, 2):
            print('   ', self.miller[i])