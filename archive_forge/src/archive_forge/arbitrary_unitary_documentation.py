import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
Compute the expected shape of the weights tensor.

        Args:
            n_wires (int): number of wires that template acts on
        