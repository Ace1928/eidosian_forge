from itertools import product
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.tensorproduct import TensorProduct, tensor_product_simp
from sympy.physics.quantum.trace import Tr
class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """

    @classmethod
    def _eval_args(cls, args):
        args = super()._eval_args(args)
        for arg in args:
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError('Each argument should be of form [state,prob] or ( state, prob )')
        return args

    def states(self):
        """Return list of all states.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()
        (|0>, |1>)

        """
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        """Return list of all probabilities.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()
        (0.5, 0.5)

        """
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        """Return specific state by index.

        Parameters
        ==========

        index : index of state to be returned

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()[1]
        |1>

        """
        state = self.args[index][0]
        return state

    def get_prob(self, index):
        """Return probability of specific state by index.

        Parameters
        ===========

        index : index of states whose probability is returned.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()[1]
        0.500000000000000

        """
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        """op will operate on each individual state.

        Parameters
        ==========

        op : Operator

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.apply_op(A)
        Density((A*|0>, 0.5),(A*|1>, 0.5))

        """
        new_args = [(op * state, prob) for state, prob in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        """Expand the density operator into an outer product format.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.doit()
        0.5*|0><0| + 0.5*|1><1|

        """
        terms = []
        for state, prob in self.args:
            state = state.expand()
            if isinstance(state, Add):
                for arg in product(state.args, repeat=2):
                    terms.append(prob * self._generate_outer_prod(arg[0], arg[1]))
            else:
                terms.append(prob * self._generate_outer_prod(state, state))
        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        c_part1, nc_part1 = arg1.args_cnc()
        c_part2, nc_part2 = arg2.args_cnc()
        if len(nc_part1) == 0 or len(nc_part2) == 0:
            raise ValueError('Atleast one-pair of Non-commutative instance required for outer product.')
        if isinstance(nc_part1[0], TensorProduct) and len(nc_part1) == 1 and (len(nc_part2) == 1):
            op = tensor_product_simp(nc_part1[0] * Dagger(nc_part2[0]))
        else:
            op = Mul(*nc_part1) * Dagger(Mul(*nc_part2))
        return Mul(*c_part1) * Mul(*c_part2) * op

    def _represent(self, **options):
        return represent(self.doit(), **options)

    def _print_operator_name_latex(self, printer, *args):
        return '\\rho'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('ρ')

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', [])
        return Tr(self.doit(), indices).doit()

    def entropy(self):
        """ Compute the entropy of a density matrix.

        Refer to density.entropy() method  for examples.
        """
        return entropy(self)