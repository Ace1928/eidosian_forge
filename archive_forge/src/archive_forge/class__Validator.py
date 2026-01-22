import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4
import cloudpickle
from fs.base import FS as FSBase
from triad import FileSystem
from tune.concepts.checkpoint import Checkpoint
from tune.concepts.flow import Monitor, Trial, TrialDecision, TrialJudge, TrialReport
class _Validator(TrialJudge):

    def __init__(self, monitor: Optional[Monitor], budgets: List[float], continuous: bool):
        super().__init__(monitor)
        self._budgets = budgets
        self._continuous = continuous
        self._reports: List[TrialReport] = []

    @property
    def reports(self) -> List[TrialReport]:
        return self._reports

    def can_accept(self, trial: Trial) -> bool:
        return True

    def get_budget(self, trial: Trial, rung: int) -> float:
        budget = self._budgets[rung] if rung < len(self._budgets) else 0.0
        self.monitor.on_get_budget(trial, rung, budget)
        return budget

    def judge(self, report: TrialReport) -> TrialDecision:
        self.monitor.on_report(report)
        self._reports.append(report)
        decision = TrialDecision(report, budget=self.get_budget(report.trial, report.rung + 1) if self._continuous else 0.0, should_checkpoint=report.rung >= len(self._budgets) if self._continuous else True)
        self.monitor.on_judge(decision)
        return decision