from typing import Any, Dict, Tuple
from pytest import raises
from tune.exceptions import TuneCompileError
from tune.concepts.flow import Trial, TrialReport
from tune.noniterative.convert import noniterative_objective, to_noniterative_objective
from tune.noniterative.objective import NonIterativeObjectiveFunc
class F4(NonIterativeObjectiveFunc):

    def run(self, t: Trial) -> TrialReport:
        return TrialReport(t, t.params['a'] - t.params['b'], params=dict(a=1), metadata=dict(c=6))