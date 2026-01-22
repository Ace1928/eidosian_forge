import copy
import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import cirq
from cirq_google.engine import calibration
from cirq_google.engine.abstract_job import AbstractJob
Returns the recorded calibration at the time when the job was created,
        from the parent Engine object.