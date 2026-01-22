from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
class ASHAJudge(TrialJudge):

    def __init__(self, schedule: List[Tuple[float, int]], always_checkpoint: bool=False, study_early_stop: Optional[Callable[[List[Any], List[RungHeap]], bool]]=None, trial_early_stop: Optional[Callable[[TrialReport, List[TrialReport], List[RungHeap]], bool]]=None, monitor: Optional[Monitor]=None):
        super().__init__(monitor=monitor)
        self._lock = SerializableRLock()
        self._data: Dict[str, _PerPartition] = {}
        self._schedule = schedule
        self._always_checkpoint = always_checkpoint
        self._study_early_stop = study_early_stop or _default_study_early_stop
        self._trial_early_stop = trial_early_stop or _default_trial_early_stop

    @property
    def schedule(self) -> List[Tuple[float, int]]:
        return self._schedule

    @property
    def always_checkpoint(self) -> bool:
        return self._always_checkpoint

    def can_accept(self, trial: Trial) -> bool:
        return self._get_judge(trial).can_accept(trial)

    def get_budget(self, trial: Trial, rung: int) -> float:
        budget = self._get_judge(trial).get_budget(trial, rung)
        self.monitor.on_get_budget(trial, rung, budget)
        return budget

    def judge(self, report: TrialReport) -> TrialDecision:
        self.monitor.on_report(report)
        decision = self._get_judge(report.trial).judge(report)
        self.monitor.on_judge(decision)
        return decision

    def _get_judge(self, trial: Trial) -> _PerPartition:
        key = to_uuid(trial.keys)
        with self._lock:
            if key not in self._data:
                self._data[key] = _PerPartition(self, trial.keys)
            return self._data[key]