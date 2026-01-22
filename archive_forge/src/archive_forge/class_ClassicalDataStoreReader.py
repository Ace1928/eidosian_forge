import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
class ClassicalDataStoreReader(abc.ABC):

    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the measurement keys in the order they were stored."""

    @property
    @abc.abstractmethod
    def records(self) -> Mapping['cirq.MeasurementKey', List[Tuple[int, ...]]]:
        """Gets the a mapping from measurement key to measurement records."""

    @property
    @abc.abstractmethod
    def channel_records(self) -> Mapping['cirq.MeasurementKey', List[int]]:
        """Gets the a mapping from measurement key to channel measurement records."""

    @abc.abstractmethod
    def get_int(self, key: 'cirq.MeasurementKey', index=-1) -> int:
        """Gets the integer corresponding to the measurement.

        The integer is determined by summing the qubit-dimensional basis value
        of each measured qubit. For example, if the measurement of qubits
        [q1, q0] produces [1, 0], then the corresponding integer is 2, the big-
        endian equivalent. If they are qutrits and the measurement is [2, 1],
        then the integer is 2 * 3 + 1 = 7.

        Args:
            key: The measurement key.
            index: If multiple measurements have the same key, the index
                argument can be used to specify which measurement to retrieve.
                Here `0` refers to the first measurement, and `-1` refers to
                the most recent.

        Raises:
            KeyError: If the key has not been used or if the index is out of
                bounds.
        """

    @abc.abstractmethod
    def get_digits(self, key: 'cirq.MeasurementKey', index=-1) -> Tuple[int, ...]:
        """Gets the values of the qubits that were measured into this key.

        For example, if the measurement of qubits [q0, q1] produces [0, 1],
        this function will return (0, 1).

        Args:
            key: The measurement key.
            index: If multiple measurements have the same key, the index
                argument can be used to specify which measurement to retrieve.
                Here `0` refers to the first measurement, and `-1` refers to
                the most recent.

        Raises:
            KeyError: If the key has not been used or if the index is out of
                bounds.
        """

    @abc.abstractmethod
    def copy(self) -> Self:
        """Creates a copy of the object."""