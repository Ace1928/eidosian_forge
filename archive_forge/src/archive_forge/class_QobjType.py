from enum import Enum, IntEnum
class QobjType(str, Enum):
    """Qobj.type allowed values."""
    QASM = 'QASM'
    PULSE = 'PULSE'