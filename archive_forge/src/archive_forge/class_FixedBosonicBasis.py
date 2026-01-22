from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class FixedBosonicBasis(BosonicBasis):
    """
    Fixed particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import FixedBosonicBasis
    >>> b = FixedBosonicBasis(2, 2)
    >>> state = b.state(1)
    >>> b
    [FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]
    >>> state
    FockStateBosonKet((1, 1))
    >>> b.index(state)
    1
    """

    def __init__(self, n_particles, n_levels):
        self.n_particles = n_particles
        self.n_levels = n_levels
        self._build_particle_locations()
        self._build_states()

    def _build_particle_locations(self):
        tup = ['i%i' % i for i in range(self.n_particles)]
        first_loop = 'for i0 in range(%i)' % self.n_levels
        other_loops = ''
        for cur, prev in zip(tup[1:], tup):
            temp = 'for %s in range(%s + 1) ' % (cur, prev)
            other_loops = other_loops + temp
        tup_string = '(%s)' % ', '.join(tup)
        list_comp = '[%s %s %s]' % (tup_string, first_loop, other_loops)
        result = eval(list_comp)
        if self.n_particles == 1:
            result = [(item,) for item in result]
        self.particle_locations = result

    def _build_states(self):
        self.basis = []
        for tuple_of_indices in self.particle_locations:
            occ_numbers = self.n_levels * [0]
            for level in tuple_of_indices:
                occ_numbers[level] += 1
            self.basis.append(FockStateBosonKet(occ_numbers))
        self.n_basis = len(self.basis)

    def index(self, state):
        """Returns the index of state in basis.

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.index(b.state(3))
        3
        """
        return self.basis.index(state)

    def state(self, i):
        """Returns the state that lies at index i of the basis

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.state(3)
        FockStateBosonKet((1, 0, 1))
        """
        return self.basis[i]

    def __getitem__(self, i):
        return self.state(i)

    def __len__(self):
        return len(self.basis)

    def __repr__(self):
        return repr(self.basis)