from abc import ABC, abstractmethod
from typing import Optional, List, Iterable, Tuple, Union, Callable
import numpy as np
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
class BaseReadoutMitigator(ABC):
    """Base readout error mitigator class."""

    @abstractmethod
    def quasi_probabilities(self, data: Counts, qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> QuasiDistribution:
        """Convert counts to a dictionary of quasi-probabilities

        Args:
            data: Counts to be mitigated.
            qubits: the physical qubits measured to obtain the counts clbits.
                If None these are assumed to be qubits [0, ..., N-1]
                for N-bit counts.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"
                is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state,
                and "mean" is the mean of non-zero quasi-probability estimates.
        """

    @abstractmethod
    def expectation_value(self, data: Counts, diagonal: Union[Callable, dict, str, np.ndarray], qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> Tuple[float, float]:
        """Calculate the expectation value of a diagonal Hermitian operator.

        Args:
            data: Counts object to be mitigated.
            diagonal: the diagonal operator. This may either be specified
                      as a string containing I,Z,0,1 characters, or as a
                      real valued 1D array_like object supplying the full diagonal,
                      or as a dictionary, or as Callable.
            qubits: the physical qubits measured to obtain the counts clbits.
                    If None these are assumed to be qubits [0, ..., N-1]
                    for N-bit counts.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            The mean and an upper bound of the standard deviation of operator
            expectation value calculated from the current counts.
        """