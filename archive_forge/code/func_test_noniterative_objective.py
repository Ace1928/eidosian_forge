import pandas as pd
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune import Space, MetricLogger
from tune.api.factory import (
from tune.concepts.dataset import TuneDataset
from tune.concepts.flow.judge import Monitor
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
from tune_optuna.optimizer import OptunaLocalOptimizer
def test_noniterative_objective():
    assert isinstance(parse_noniterative_objective(_nobjective), NonIterativeObjectiveFunc)
    with raises(TuneCompileError):
        parse_noniterative_objective('x')

    @parse_noniterative_objective.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return to_noniterative_objective(_nobjective)
    assert isinstance(parse_noniterative_objective(_DUMMY), NonIterativeObjectiveFunc)