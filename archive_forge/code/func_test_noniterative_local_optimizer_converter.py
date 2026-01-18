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
def test_noniterative_local_optimizer_converter():
    assert isinstance(parse_noniterative_local_optimizer(OptunaLocalOptimizer(0)), NonIterativeObjectiveLocalOptimizer)
    with raises(TuneCompileError):
        parse_noniterative_local_optimizer('x')

    @parse_noniterative_local_optimizer.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return OptunaLocalOptimizer(0)
    assert isinstance(parse_noniterative_local_optimizer(_DUMMY), NonIterativeObjectiveLocalOptimizer)