from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
class NumpyWavefunctionSimulator(AbstractQuantumSimulator):

    def __init__(self, n_qubits: int, rs: Optional[RandomState]=None):
        """
        A wavefunction simulator that uses numpy's tensordot or einsum to update a state vector

        Please consider using
        :py:class:`PyQVM(..., quantum_simulator_type=NumpyWavefunctionSimulator)` rather
        than using this class directly.

        This class uses a n_qubit-dim ndarray to store wavefunction
        amplitudes. The array is indexed into with a tuple of n_qubits 1's and 0's, with
        qubit 0 as the leftmost bit. This is the opposite convention of the Rigetti Lisp QVM.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (should be shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic. A value of ``None`` disallows doing anything stochastic.
        """
        super().__init__(n_qubits=n_qubits, rs=rs)
        self.n_qubits = n_qubits
        self.rs = rs
        self.wf = np.zeros((2,) * n_qubits, dtype=np.complex128)
        self.wf[(0,) * n_qubits] = complex(1.0, 0)

    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """
        Sample bitstrings from the distribution defined by the wavefunction.

        Qubit 0 is at ``out[:, 0]``.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
        if self.rs is None:
            raise ValueError('You have tried to perform a stochastic operation without setting the random state of the simulator. Might I suggest using a PyQVM object?')
        probabilities = np.abs(self.wf.reshape(-1)) ** 2
        possible_bitstrings = all_bitstrings(self.n_qubits)
        inds = self.rs.choice(2 ** self.n_qubits, n_samples, p=probabilities)
        return possible_bitstrings[inds, :]

    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit, collapse the wavefunction, and return the measurement result.

        :param qubit: Index of the qubit to measure.
        :return: measured bit
        """
        if self.rs is None:
            raise ValueError('You have tried to perform a stochastic operation without setting the random state of the simulator. Might I suggest using a PyQVM object?')
        measurement_probs = get_measure_probabilities(self.wf, qubit)
        measured_bit = int(np.argmax(self.rs.uniform() < np.cumsum(measurement_probs)))
        other_bit = (measured_bit + 1) % 2
        other_bit_indices = (slice(None),) * qubit + (other_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
        self.wf[other_bit_indices] = 0
        meas_bit_indices = (slice(None),) * qubit + (measured_bit,) + (slice(None),) * (self.n_qubits - qubit - 1)
        self.wf[meas_bit_indices] /= np.sqrt(measurement_probs[measured_bit])
        return measured_bit

    def do_gate(self, gate: Gate) -> 'NumpyWavefunctionSimulator':
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """
        gate_matrix, qubit_inds = _get_gate_tensor_and_qubits(gate=gate)
        self.wf = targeted_tensordot(gate=gate_matrix, wf=self.wf, wf_target_inds=qubit_inds)
        return self

    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> 'NumpyWavefunctionSimulator':
        """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """
        tensor = np.reshape(matrix, (2,) * len(qubits) * 2)
        self.wf = targeted_tensordot(gate=tensor, wf=self.wf, wf_target_inds=qubits)
        return self

    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> float:
        """
        Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """
        if not isinstance(operator, PauliSum):
            operator = PauliSum([operator])
        return sum((_term_expectation(self.wf, term) for term in operator))

    def reset(self) -> 'NumpyWavefunctionSimulator':
        """
        Reset the wavefunction to the ``|000...00>`` state.

        :return: ``self`` to support method chaining.
        """
        self.wf.fill(0)
        self.wf[(0,) * self.n_qubits] = complex(1.0, 0)
        return self

    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: List[int]) -> 'AbstractQuantumSimulator':
        raise NotImplementedError('The numpy simulator cannot handle noise')