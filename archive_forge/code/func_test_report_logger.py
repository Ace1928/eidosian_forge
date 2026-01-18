import copy
from time import sleep
import numpy as np
import pandas as pd
from tune.concepts.flow import (
from tune.concepts.space import Rand, TuningParametersTemplate
import cloudpickle
def test_report_logger():

    class Mock(TrialReportLogger):

        def __init__(self, new_best_only: bool=False):
            super().__init__(new_best_only=new_best_only)
            self._values = []

        def log(self, report: TrialReport) -> None:
            return self._values.append(report)

        @property
        def records(self):
            return self._values
    t1 = Trial('a', dict(a=1, b=2), keys=['x', 'y'])
    r1 = TrialReport(t1, 0.8, sort_metric=-0.8)
    t2 = Trial('b', dict(a=11, b=12), keys=['xx', 'y'])
    r2 = TrialReport(t2, 0.7, sort_metric=-0.7)
    t3 = Trial('c', dict(a=10, b=20), keys=['x', 'y'])
    r3 = TrialReport(t3, 0.9, sort_metric=-0.9)
    b = Mock()
    assert 0 == len(b.records)
    assert b.best is None
    assert b.on_report(r1)
    assert b.on_report(r3)
    assert r3 is b.best
    assert 2 == len(b.records)
    b = Mock()
    assert b.on_report(r3)
    assert not b.on_report(r1)
    assert r3 is b.best
    assert 2 == len(b.records)
    b = Mock(new_best_only=True)
    assert b.on_report(r3)
    assert not b.on_report(r1)
    assert r3 is b.best
    assert 1 == len(b.records)