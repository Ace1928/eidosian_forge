import enum
import itertools
from typing import Dict, Sequence, Tuple, Union, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate, SwapPermutationGate
A swap network that acquaints qubits in one half with qubits in the
    other.


    Acts on 2k qubits, acquainting some of the first k qubits with some of the
    latter k. May have the effect permuting the qubits within each half.

    Possible subgraphs include:
        MATCHING: acquaints qubit 1 with qubit (2k - 1), qubit 2 with qubit
            (2k- 2), and so on through qubit k with qubit k + 1.
        COMPLETE: acquaints each of qubits 1 through k with each of qubits k +
            1 through 2k.

    Args:
        part_size: The number of qubits in each half.
        subgraph: The bipartite subgraph of pairs of qubits to acquaint.
        swap_gate: The gate used to swap logical indices.

    Attributes:
        part_size: See above.
        subgraph: See above.
        swap_gate: See above.
    