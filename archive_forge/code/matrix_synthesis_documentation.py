from qiskit.circuit.quantumcircuit import QuantumCircuit
from .evolution_synthesis import EvolutionSynthesis
Exact operator evolution via matrix exponentiation and unitary synthesis.

    This class synthesis the exponential of operators by calculating their exponentially-sized
    matrix representation and using exact matrix exponentiation followed by unitary synthesis
    to obtain a circuit. This process is not scalable and serves as comparison or benchmark
    for small systems.
    