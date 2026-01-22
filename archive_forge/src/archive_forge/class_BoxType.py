from enum import Enum
from typing import NamedTuple, List, Union, NewType, Tuple, Dict
from qiskit import circuit
class BoxType(str, Enum):
    """Box type.

    SCHED_GATE: Box that represents occupation time by gate.
    DELAY: Box associated with delay.
    TIMELINE: Box that represents time slot of a bit.
    """
    SCHED_GATE = 'Box.ScheduledGate'
    DELAY = 'Box.Delay'
    TIMELINE = 'Box.Timeline'