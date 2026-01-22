import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
class MeasurementType(enum.IntEnum):
    """Type of a measurement, whether a measurement or channel.

    This determines how the results of a measurement are stored
    as classical data in a `ClassicalDataStoreRegister`.
    `MEASUREMENT` represent measurements of a `Cirq.Qid`
    (for instance, a qubit or qudit).  A `CHANNEL` represents
    the measurement of a channel, such as the set of Kraus
    operators.  In this case, the data stored is the integer
    index of the channel measured.
    """
    MEASUREMENT = 1
    CHANNEL = 2

    def __repr__(self):
        return f'cirq.MeasurementType.{self.name}'