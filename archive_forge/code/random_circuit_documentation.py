from typing import List, Union, Sequence, Dict, Optional, TYPE_CHECKING
from cirq import circuits, ops, value
from cirq.ops import Qid
from cirq._doc import document
Creates a random two qubit circuit with the given number of CNOTs.

    The resulting circuit will have `num_cnots` number of CNOTs that will be
    surrounded by random `PhasedXPowGate` instances on both qubits.

    Args:
         num_czs: the number of CNOTs to be guaranteed in the circuit
         q0: the first qubit the circuit should operate on
         q1: the second qubit the circuit should operate on
         random_state: an optional random seed
    Returns:
         the random two qubit circuit
    