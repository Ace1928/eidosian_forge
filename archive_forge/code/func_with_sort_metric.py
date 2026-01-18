import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
def with_sort_metric(self, sort_metric: Any) -> 'TrialReport':
    """Construct a new report object with the new ``sort_metric``

        :param sort_metric: new sort_metric
        :return: a new object with the updated value
        """
    t = self.copy()
    t._sort_metric = float(sort_metric)
    return t