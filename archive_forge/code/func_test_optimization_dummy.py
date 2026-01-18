from typing import Any, Callable, Dict, List, Tuple
from unittest import TestCase
from triad import SerializableRLock
from tune import (
from tune._utils import assert_close
from tune.noniterative.objective import validate_noniterative_objective
def test_optimization_dummy(self):
    params = dict(a=1, b=2, c=3)
    trial = Trial('a', params, metadata={})
    o = self.make_optimizer(max_iter=5)

    @noniterative_objective
    def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
        return (a ** 2 + b ** 2 + c, dict(a='x'))

    def v(report):
        assert 1 == report.params.simple_value['a']
        assert 2 == report.params.simple_value['b']
        assert 3 == report.params.simple_value['c']
        assert report.metric == 8
        assert 'x' == report.metadata['a']
    validate_noniterative_objective(objective, trial, v, optimizer=o)