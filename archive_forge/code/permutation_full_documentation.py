from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import (
Synthesize a permutation circuit for a fully-connected
    architecture using the Alon, Chung, Graham method.

    This produces a quantum circuit of depth 2 (measured in the number of SWAPs).

    This implementation is based on the Proposition 4.1 in reference [1] with
    the detailed proof given in Theorem 2 in reference [2]

    Args:
        pattern: Permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        The synthesized quantum circuit.

    References:
        1. N. Alon, F. R. K. Chung, and R. L. Graham.
           *Routing Permutations on Graphs Via Matchings.*,
           Proceedings of the Twenty-Fifth Annual ACM Symposium on Theory of Computing(1993).
           Pages 583–591.
           `(Extended abstract) 10.1145/167088.167239 <https://doi.org/10.1145/167088.167239>`_
        2. N. Alon, F. R. K. Chung, and R. L. Graham.
           *Routing Permutations on Graphs Via Matchings.*,
           `(Full paper) <https://www.cs.tau.ac.il/~nogaa/PDFS/r.pdf>`_
    