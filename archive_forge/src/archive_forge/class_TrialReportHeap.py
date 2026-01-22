import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
class TrialReportHeap:

    class _Wrapper:

        def __init__(self, report: TrialReport, min_heap: bool):
            self.report = report
            self.min_heap = min_heap

        def __lt__(self, other: 'TrialReportHeap._Wrapper') -> bool:
            k1 = (self.report.sort_metric, self.report.cost, self.report.rung)
            k2 = (other.report.sort_metric, other.report.cost, other.report.rung)
            return k1 < k2 if self.min_heap else k2 < k1

        @property
        def trial_id(self) -> str:
            return self.report.trial_id

    def __init__(self, min_heap: bool):
        self._data: List[TrialReportHeap._Wrapper] = []
        self._ids: Set[str] = set()
        self._min_heap = min_heap

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, tid: str) -> bool:
        return tid in self._ids

    def values(self) -> Iterable[TrialReport]:
        for x in self._data:
            yield x.report

    def push(self, report: TrialReport) -> None:
        w = TrialReportHeap._Wrapper(report, self._min_heap)
        if w.trial_id in self._ids:
            self._data = [x if x.trial_id != w.trial_id else w for x in self._data]
            heapq.heapify(self._data)
        else:
            self._ids.add(w.trial_id)
            heapq.heappush(self._data, w)

    def pop(self) -> TrialReport:
        w = heapq.heappop(self._data)
        self._ids.remove(w.trial_id)
        return w.report