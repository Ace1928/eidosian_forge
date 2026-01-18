from typing import TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.qcircuit.qcircuit_diagram_info import (
Returns a QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to represent in latex.
        qubit_order: Determines the order of qubit wires in the diagram.

    Returns:
        Latex code for the diagram.
    