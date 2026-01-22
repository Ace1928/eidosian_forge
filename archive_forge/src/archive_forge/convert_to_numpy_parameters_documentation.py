import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumScript
Transforms a circuit to one with purely numpy parameters.

    Args:
        circuit (QuantumScript): a circuit with parameters of any interface

    Returns:
        QuantumScript: A circuit with purely numpy parameters

    .. seealso::

        :class:`pennylane.tape.Unwrap` modifies a :class:`~.pennylane.tape.QuantumScript` in place instead of creating
        a new class. It will also set all parameters on the circuit, not just ones that need to be unwrapped.

    >>> ops = [qml.S(0), qml.RX(torch.tensor(0.1234), 0)]
    >>> measurements = [qml.state(), qml.expval(qml.Hermitian(torch.eye(2), 0))]
    >>> circuit = qml.tape.QuantumScript(ops, measurements )
    >>> new_circuit = convert_to_numpy_parameters(circuit)
    >>> new_circuit.circuit
    [S(wires=[0]),
    RX(0.1234000027179718, wires=[0]),
    state(wires=[]),
    expval(Hermitian(array([[1., 0.],
            [0., 1.]], dtype=float32), wires=[0]))]

    If the component's data does not need to be transformed, it is left uncopied.

    >>> circuit[0] is new_circuit[0]
    True
    >>> circuit[1] is new_circuit[1]
    False
    >>> circuit[2] is new_circuit[2]
    True
    >>> circuit[3] is new_circuit[3]
    False

    