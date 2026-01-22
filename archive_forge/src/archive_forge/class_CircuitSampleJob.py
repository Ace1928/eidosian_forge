import abc
from typing import Any, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
import duet
import numpy as np
from cirq import study, value
@value.value_equality(unhashable=True)
class CircuitSampleJob:
    """Describes a sampling task."""

    def __init__(self, circuit: 'cirq.AbstractCircuit', *, repetitions: int, tag: Any=None):
        """Inits CircuitSampleJob.

        Args:
            circuit: The circuit to sample from.
            repetitions: How many times to sample the circuit.
            tag: An arbitrary value associated with the job. This value is used
                so that when a job completes and is handed back, it is possible
                to tell what the job was for. For example, the key could be a
                string like "main_run" or "calibration_run", or it could be set
                to the component of the Hamiltonian (e.g. a PauliString) that
                the circuit is supposed to be helping to estimate.
        """
        self.circuit = circuit
        self.repetitions = repetitions
        self.tag = tag

    def _value_equality_values_(self) -> Any:
        return (self.circuit, self.repetitions, self.tag)

    def __repr__(self) -> str:
        return f'cirq.CircuitSampleJob(tag={self.tag!r}, repetitions={self.repetitions!r}, circuit={self.circuit!r})'