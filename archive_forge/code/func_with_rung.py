import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
def with_rung(self, rung: int) -> 'TrialReport':
    """Construct a new report object with the new ``rung``

        :param rung: new rung
        :return: a new object with the updated value
        """
    t = self.copy()
    t._rung = rung
    return t