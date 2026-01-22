from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array

        gate dcx a, b { cx a, b; cx b, a; }
        