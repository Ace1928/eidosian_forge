import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        