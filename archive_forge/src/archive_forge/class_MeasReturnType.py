from enum import Enum, IntEnum
class MeasReturnType(str, Enum):
    """PulseQobjConfig meas_return allowed values."""
    AVERAGE = 'avg'
    SINGLE = 'single'