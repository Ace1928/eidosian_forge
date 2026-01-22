import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
class _NamedOneQubitState(metaclass=abc.ABCMeta):
    """Abstract class representing a one-qubit state of note."""

    def on(self, qubit: 'cirq.Qid') -> 'ProductState':
        """Associates one qubit with this named state.

        The returned object is a ProductState of length 1.
        """
        return ProductState({qubit: self})

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)

    @abc.abstractmethod
    def state_vector(self) -> np.ndarray:
        """Return a state vector representation of the named state."""

    def projector(self) -> np.ndarray:
        """Return |s⟩⟨s| as a matrix for the named state."""
        vec = self.state_vector()[:, np.newaxis]
        return vec @ vec.conj().T