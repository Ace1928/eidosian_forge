from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
class SimpleNonIterativeStopper(NonIterativeStopper):

    def __init__(self, partition_should_stop: Callable[[TrialReport, bool, List[TrialReport]], bool], log_best_only: bool=False):
        super().__init__(log_best_only=log_best_only)
        self._partition_should_stop = partition_should_stop
        self._stopped: Set[str] = set()

    def should_stop(self, trial: Trial) -> bool:
        key = str(trial.keys)
        with self._lock:
            return key in self._stopped

    def on_report(self, report: TrialReport) -> bool:
        updated = super().on_report(report)
        key = str(report.trial.keys)
        with self._lock:
            if key not in self._stopped:
                if self._partition_should_stop(report, updated, self.get_reports(report.trial)):
                    self._stopped.add(key)
        return updated