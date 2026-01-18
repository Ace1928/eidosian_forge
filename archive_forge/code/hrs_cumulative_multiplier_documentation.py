from typing import Optional
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from .multiplier import Multiplier

        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            num_result_qubits: The number of result qubits to limit the output to.
                If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed
                to limit the output to the specified number of qubits. Default
                value is ``2 * num_state_qubits`` to represent any possible
                result from the multiplication of the two inputs.
            adder: Half adder circuit to be used for performing multiplication. The
                CDKMRippleCarryAdder is used as default if no adder is provided.
            name: The name of the circuit object.
        Raises:
            NotImplementedError: If ``num_result_qubits`` is not default and a custom adder is provided.
        