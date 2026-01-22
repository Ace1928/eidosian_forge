from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Barrier
class UnitaryOverlap(QuantumCircuit):
    """Circuit that returns the overlap between two unitaries :math:`U_2^{\\dag} U_1`.

    The input quantum circuits must represent unitary operations, since they must be invertible.
    If the inputs will have parameters, they are replaced by :class:`.ParameterVector`\\s with
    names `"p1"` (for circuit ``unitary1``) and `"p2"` (for circuit ``unitary_2``) in the output
    circuit.

    This circuit is usually employed in computing the fidelity::

        .. math::

            \\left|\\langle 0| U_2^{\\dag} U_1|0\\rangle\\right|^{2}

    by computing the probability of being in the all-zeros bit-string, or equivalently,
    the expectation value of projector :math:`|0\\rangle\\langle 0|`.

    Example::

        import numpy as np
        from qiskit.circuit.library import EfficientSU2, UnitaryOverlap
        from qiskit.primitives import Sampler

        # get two circuit to prepare states of which we comput the overlap
        circuit = EfficientSU2(2, reps=1)
        unitary1 = circuit.assign_parameters(np.random.random(circuit.num_parameters))
        unitary2 = circuit.assign_parameters(np.random.random(circuit.num_parameters))

        # create the overlap circuit
        overlap = UnitaryOverap(unitary1, unitary2)

        # sample from the overlap
        sampler = Sampler(options={"shots": 100})
        result = sampler.run(overlap).result()

        # the fidelity is the probability to measure 0
        fidelity = result.quasi_dists[0].get(0, 0)

    """

    def __init__(self, unitary1: QuantumCircuit, unitary2: QuantumCircuit, prefix1='p1', prefix2='p2'):
        """
        Args:
            unitary1: Unitary acting on the ket vector.
            unitary2: Unitary whose inverse operates on the bra vector.
            prefix1: The name of the parameter vector associated to ``unitary1``,
                if it is parameterized. Defaults to ``"p1"``.
            prefix2: The name of the parameter vector associated to ``unitary2``,
                if it is parameterized. Defaults to ``"p2"``.

        Raises:
            CircuitError: Number of qubits in ``unitary1`` and ``unitary2`` does not match.
            CircuitError: Inputs contain measurements and/or resets.
        """
        if unitary1.num_qubits != unitary2.num_qubits:
            raise CircuitError(f'Number of qubits in unitaries does not match: {unitary1.num_qubits} != {unitary2.num_qubits}.')
        unitaries = [unitary1, unitary2]
        for unitary in unitaries:
            _check_unitary(unitary)
        for i, prefix in enumerate([prefix1, prefix2]):
            if unitaries[i].num_parameters > 0:
                new_params = ParameterVector(prefix, unitaries[i].num_parameters)
                unitaries[i] = unitaries[i].assign_parameters(new_params)
        super().__init__(unitaries[0].num_qubits, name='UnitaryOverlap')
        self.compose(unitaries[0], inplace=True)
        self.compose(unitaries[1].inverse(), inplace=True)