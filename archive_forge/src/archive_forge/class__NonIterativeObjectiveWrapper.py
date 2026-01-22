from typing import Optional
from triad import FileSystem
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.objective import NonIterativeObjectiveFunc
from tune.concepts.flow import Trial, TrialDecision, TrialJudge, TrialReport
class _NonIterativeObjectiveWrapper(NonIterativeObjectiveFunc):

    def __init__(self, func: IterativeObjectiveFunc, checkpoint_path: str, budget: float):
        super().__init__()
        self._budget = budget
        self._func = func
        self._checkpoint_path = checkpoint_path

    def generate_sort_metric(self, value: float) -> float:
        return self._func.generate_sort_metric(value)

    def run(self, trial: Trial) -> TrialReport:
        judge = _NonIterativeJudgeWrapper(self._budget)
        base_fs = FileSystem()
        fs = base_fs.makedirs(self._checkpoint_path, recreate=True)
        self._func = self._func.copy()
        self._func.run(trial, judge=judge, checkpoint_basedir_fs=fs)
        return judge.report