import collections
from typing import cast, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from cirq import circuits, ops, transformers
from cirq.contrib.acquaintance.gates import SwapNetworkGate, AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.devices import get_acquaintance_size
from cirq.contrib.acquaintance.permutation import PermutationGate
class ExposeAcquaintanceGates:
    """Decomposes permutation gates that provide acquaintance opportunities."""

    def __init__(self):
        self.no_decomp = lambda op: not get_acquaintance_size(op) or isinstance(op.gate, AcquaintanceOpportunityGate)

    def optimize_circuit(self, circuit: 'cirq.Circuit') -> None:
        circuit._moments = [*transformers.expand_composite(circuit, no_decomp=self.no_decomp)]

    def __call__(self, circuit: 'cirq.Circuit') -> None:
        self.optimize_circuit(circuit)