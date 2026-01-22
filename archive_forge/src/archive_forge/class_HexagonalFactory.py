from ase.lattice.triclinic import TriclinicFactory
class HexagonalFactory(TriclinicFactory):
    """A factory for creating simple hexagonal lattices."""
    xtal_name = 'hexagonal'

    def make_crystal_basis(self):
        """Make the basis matrix for the crystal and system unit cells."""
        if isinstance(self.latticeconstant, type({})):
            self.latticeconstant['alpha'] = 90
            self.latticeconstant['beta'] = 90
            self.latticeconstant['gamma'] = 120
            self.latticeconstant['b/a'] = 1.0
        elif len(self.latticeconstant) == 2:
            a, c = self.latticeconstant
            self.latticeconstant = (a, a, c, 90, 90, 120)
        else:
            raise ValueError('Improper lattice constants for hexagonal crystal.')
        TriclinicFactory.make_crystal_basis(self)

    def find_directions(self, directions, miller):
        """Find missing directions and miller indices from the specified ones.

        Also handles the conversion of hexagonal-style 4-index notation to
        the normal 3-index notation.
        """
        directions = list(directions)
        miller = list(miller)
        if miller != [None, None, None]:
            raise NotImplementedError('Specifying Miller indices of surfaces currently broken for hexagonal crystals.')
        for obj in (directions, miller):
            for i in range(3):
                if obj[i] is not None:
                    a, b, c, d = obj[i]
                    if a + b + c != 0:
                        raise ValueError(('(%d,%d,%d,%d) is not a valid hexagonal Miller ' + 'index, as the sum of the first three numbers ' + 'should be zero.') % (a, b, c, d))
                    x = 4 * a + 2 * b
                    y = 2 * a + 4 * b
                    z = 3 * d
                    obj[i] = (x, y, z)
        TriclinicFactory.find_directions(self, directions, miller)

    def print_directions_and_miller(self, txt=''):
        """Print direction vectors and Miller indices."""
        print('Direction vectors of unit cell%s:' % (txt,))
        for i in (0, 1, 2):
            self.print_four_vector('[]', self.directions[i])
        print('Miller indices of surfaces%s:' % (txt,))
        for i in (0, 1, 2):
            self.print_four_vector('()', self.miller[i])

    def print_four_vector(self, bracket, numbers):
        bra, ket = bracket
        x, y, z = numbers
        a = 2 * x - y
        b = -x + 2 * y
        c = -x - y
        d = 2 * z
        print('   %s%d, %d, %d%s  ~  %s%d, %d, %d, %d%s' % (bra, x, y, z, ket, bra, a, b, c, d, ket))