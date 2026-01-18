from typing import Any, Callable, Dict, List, Tuple
from unittest import TestCase
from triad import SerializableRLock
from tune import (
from tune._utils import assert_close
from tune.noniterative.objective import validate_noniterative_objective
def test_optimization_nested_param(self):
    params = dict(a=dict(x=Rand(-10.0, 10.0)), b=[RandInt(-100, 100)], c=[2.0])
    trial = Trial('a', params, metadata={})
    o = self.make_optimizer(max_iter=200)

    @noniterative_objective
    def objective(a, b, c) -> Tuple[float, Dict[str, Any]]:
        return (a['x'] ** 2 + b[0] ** 2 + c[0], dict(a='x'))

    def v(report):
        print(report.metric)
        assert report.metric < 7
        assert report.params.simple_value['a']['x'] ** 2 < 2
        assert report.params.simple_value['b'][0] ** 2 < 2
        assert 2.0 == report.params.simple_value['c'][0]
        assert 'x' == report.metadata['a']
    validate_noniterative_objective(objective, trial, v, optimizer=o)