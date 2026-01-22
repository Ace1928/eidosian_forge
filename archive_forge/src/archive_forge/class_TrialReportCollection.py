from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
class TrialReportCollection(TrialReportLogger):

    def __init__(self, new_best_only: bool=False):
        super().__init__(new_best_only=new_best_only)
        self._reports: List[TrialReport] = []

    def log(self, report: TrialReport) -> None:
        self._reports.append(report.reset_log_time())

    @property
    def reports(self) -> List[TrialReport]:
        with self._lock:
            return list(self._reports)