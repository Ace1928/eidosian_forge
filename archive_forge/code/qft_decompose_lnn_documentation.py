import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear_phase.cz_depth_lnn import _append_cx_stage1, _append_cx_stage2
Synthesis of a QFT circuit for a linear nearest neighbor connectivity.
    Based on Fig 2.b in Fowler et al. [1].

    Note that this method *reverts* the order of qubits in the circuit,
    compared to the original :class:`.QFT` code.
    Hence, the default value of the ``do_swaps`` parameter is ``True``
    since it produces a circuit with fewer CX gates.

    Args:
        num_qubits: The number of qubits on which the QFT acts.
        approximation_degree: The degree of approximation (0 for no approximation).
        do_swaps: Whether to include the final swaps in the QFT.

    Returns:
        A circuit implementation of the QFT circuit.

    References:
        1. A. G. Fowler, S. J. Devitt, and L. C. L. Hollenberg,
           *Implementation of Shor's algorithm on a linear nearest neighbour qubit array*,
           Quantum Info. Comput. 4, 4 (July 2004), 237–251.
           `arXiv:quant-ph/0402196 [quant-ph] <https://arxiv.org/abs/quant-ph/0402196>`_
    