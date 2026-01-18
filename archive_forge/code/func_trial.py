import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
@property
def trial(self) -> Trial:
    """The original trial sent to the objective"""
    return self._trial