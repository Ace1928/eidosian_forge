import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class ActiveReset(Message):
    """
    An active reset control sequence consisting of a repeated
      sequence of a measurement block and a feedback block conditional on the
      outcome of a specific measurement bit.  Regardless of the measurement
      outcomes the total duration of the control sequence is [attempts x
      (measurement_duration + feedback_duration)].  The total
      measurement_duration must be chosen to allow for enough time after any
      Capture commands for the measurement bit to propagate back to the gate
      cards that are actuating the feedback.
    """
    time: float
    'Time at which the ActiveReset begins in [seconds].'
    measurement_duration: float
    'The duration of measurement block in [seconds]. The\n          measurement bit is expected to have arrived on the QGS after\n          this time relative to the overall start of the ActiveReset block.'
    feedback_duration: float
    'The duration of feedback block in [seconds]'
    measurement_bit: int
    'The address of the readout bit to condition the\n          feedback on.  The bit is first accessed after measurement_duration\n          has elapsed.'
    attempts: int = 3
    'The number of times to repeat the active reset sequence.'
    measurement_instructions: List[Dict] = field(default_factory=list)
    'The ordered sequence of scheduled measurement\n          instructions.'
    apply_feedback_when: bool = True
    'Apply the feedback when the measurement_bit equals the\n          value of this flag.'
    feedback_instructions: List[Dict] = field(default_factory=list)
    'The ordered sequence of scheduled feedback instructions.'