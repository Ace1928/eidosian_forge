import pennylane as qml
Computes the tapes necessary to get the gradient of a tape with respect to
    a Hamiltonian observable's coefficients.

    Args:
        tape (qml.tape.QuantumTape): tape with a single Hamiltonian expectation as measurement
        idx (int): index of parameter that we differentiate with respect to
    