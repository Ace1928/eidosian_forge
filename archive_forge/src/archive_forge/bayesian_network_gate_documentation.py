import math
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
from sympy.combinatorics import GrayCode
from cirq import value
from cirq.ops import common_gates, pauli_gates, raw_types
Builds a BayesianNetworkGate.

        The network is specified by the two types of probabilitites: The probabilitites for the
        independent variables, and the probabilitites for the dependent ones.

        For example, we could have two independent variables, q0 and q1, and one dependent variable,
        q2. The independent variables could be defined as p(q0 = 1) and p(q1 = 1). The dependence
        could be defined as p(q2 = 1 | q0, q1) for the four values that (q0, q1) can take.

        In this case, the input arguments would be:
        init_prob = [
            ('q0', 0.123)   # Indicates that p(q0 = 1) = 0.123
            ('q1', 0.456)   # Indicates that p(q1 = 1) = 0.456
            ('q2', None)    # Indicates that q2 is a dependent variable
        ]
        arc_probs = [
            ('q2', ('q0', 'q1'), [0.1, 0.2, 0.3, 0.4])
                # Indicates that p(q2 = 1 | q0 = 0 and q1 = 0) = 0.1
                # Indicates that p(q2 = 1 | q0 = 0 and q1 = 1) = 0.2
                # Indicates that p(q2 = 1 | q0 = 1 and q1 = 0) = 0.3
                # Indicates that p(q2 = 1 | q0 = 1 and q1 = 1) = 0.4
        ]

        By convention, all the probabilties are for the variable being equal to 1 and the
        probability of being equal to zero can be inferred. In the example above, we thus have:
        p(q2 = 0 | q0 = 1 and q1 = 0) = 1.0 - p(q2 = 1 | q0 = 1 and q1 = 0) = 0.7

        Note that there is NO checking that the chain of probability creates a directed acyclic
        graph. In particular, the order of the elements in arc_probs matters. Also, if you want to
        specify the dependent probabilities outside of this gate, you can mark all the variables as
        dependent in init_probs.

        init_prob: A list of tuples, each representing a single variable. The first element of the
            tuples is a string representing the name of the variable. The second element of the
            tuples is either None for dependent variables, or a float representing a probability.

        arc_probs: A list of tuples, each representing a dependence. The first element of the tuples
            is a string representing the name of the variable. The second element of the tuples is
            itself a tuple of n strings, representing the dependence. The third element of the
            tuples is a list of 2**n floats, each representing the probabilities.

        Raises:
            ValueError: If the probabilities are not in [0, 1], or an incorrect number of
            probability is specified, or if the parameter names are no passed as a tuple.
        