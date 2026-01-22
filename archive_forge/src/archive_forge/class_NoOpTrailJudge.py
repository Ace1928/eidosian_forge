from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
class NoOpTrailJudge(TrialJudge):

    def can_accept(self, trial: Trial) -> bool:
        return True

    def get_budget(self, trial: Trial, rung: int) -> float:
        return 0.0

    def judge(self, report: TrialReport) -> TrialDecision:
        self.monitor.on_report(report)
        return TrialDecision(report, 0.0, False)