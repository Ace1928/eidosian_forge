import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
class ClassicalDataStore(ClassicalDataStoreReader, abc.ABC):

    @abc.abstractmethod
    def record_measurement(self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']):
        """Records a measurement.

        Args:
            key: The measurement key to hold the measurement.
            measurement: The measurement result.
            qubits: The qubits that were measured.

        Raises:
            ValueError: If the measurement shape does not match the qubits
                measured or if the measurement key was already used.
        """

    @abc.abstractmethod
    def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
        """Records a channel measurement.

        Args:
            key: The measurement key to hold the measurement.
            measurement: The measurement result.

        Raises:
            ValueError: If the measurement key was already used.
        """