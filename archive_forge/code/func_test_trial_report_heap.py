import copy
from time import sleep
import numpy as np
import pandas as pd
from tune.concepts.flow import (
from tune.concepts.space import Rand, TuningParametersTemplate
import cloudpickle
def test_trial_report_heap():
    t1 = Trial('a', {})
    r1 = TrialReport(t1, 0.1)
    t2 = Trial('b', {})
    r2 = TrialReport(t2, 0.2)
    t3 = Trial('c', {})
    r3 = TrialReport(t3, 0.3)
    r4 = TrialReport(t3, -0.3)
    h = TrialReportHeap(min_heap=True)
    for r in [r1, r2, r3, r4]:
        h.push(r)
    assert 'a' in h
    assert 'x' not in h
    for r in [r4, r1, r2]:
        assert h.pop() is r
    assert 0 == len(h)
    h = TrialReportHeap(min_heap=False)
    for r in [r1, r2, r3, r4]:
        h.push(r)
    for r in [r1, r2, r4]:
        assert r in list(h.values())
    for r in [r2, r1, r4]:
        assert h.pop() is r
    assert 0 == len(h)
    r5 = TrialReport(t1, metric=0.1, sort_metric=-0.1)
    r6 = TrialReport(t2, metric=0.2, sort_metric=-0.2)
    r7 = TrialReport(t3, metric=0.3, sort_metric=-0.3)
    h = TrialReportHeap(min_heap=True)
    for r in [r7, r6, r5]:
        h.push(r)
    for r in [r7, r6, r5]:
        assert h.pop() is r
    assert 0 == len(h)
    r5 = TrialReport(t1, metric=0.1, cost=0.2, rung=5)
    r6 = TrialReport(t2, metric=0.1, cost=0.3, rung=5)
    r7 = TrialReport(t3, metric=0.1, cost=0.3, rung=6)
    h = TrialReportHeap(min_heap=True)
    for r in [r7, r6, r5]:
        h.push(r)
    for r in [r5, r6, r7]:
        assert h.pop() is r
    assert 0 == len(h)
    r8 = TrialReport(t1, metric=0.1, cost=0.3, rung=6)
    r9 = TrialReport(t2, metric=0.1, cost=0.3, rung=6)
    h = TrialReportHeap(min_heap=False)
    for r in [r8, r9]:
        h.push(r)
    for r in [r8, r9]:
        assert h.pop() is r
    assert 0 == len(h)