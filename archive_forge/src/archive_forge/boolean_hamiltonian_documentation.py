import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
Builds a BooleanHamiltonianGate.

        For each element of a sequence of Boolean expressions, the code first transforms it into a
        polynomial of Pauli Zs that represent that particular expression. Then, we sum all the
        polynomials, thus making a function that goes from a series to Boolean inputs to an integer
        that is the number of Boolean expressions that are true.

        For example, if we were using this gate for the unweighted max-cut problem that is typically
        used to demonstrate the QAOA algorithm, there would be one Boolean expression per edge. Each
        Boolean expression would be true iff the vertices on that are in different cuts (i.e. it's)
        an XOR.

        Then, we compute exp(-j * theta * polynomial), which is unitary because the polynomial is
        Hermitian.

        Args:
            parameter_names: The names of the inputs to the expressions.
            boolean_strs: The list of Sympy-parsable Boolean expressions.
            theta: The evolution time (angle) for the Hamiltonian
        