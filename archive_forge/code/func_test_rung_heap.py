import math
from typing import Any, Dict, Iterable
from fugue import FugueWorkflow
from tune import optimize_by_continuous_asha
from tune.constants import TUNE_REPORT_METRIC
from tune.concepts.dataset import TuneDatasetBuilder
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.objective import IterativeObjectiveFunc
from tune.concepts.space import Grid, Space
from tune.concepts.flow import Monitor, Trial, TrialReport
def test_rung_heap():
    h = RungHeap(2)
    assert 2 == h.capacity
    assert not h.full
    assert math.isnan(h.best)
    assert 0 == len(h)
    assert [] == h.bests
    assert h.push(rp('a', 1))
    assert h.push(rp('b', 0.1))
    assert h.push(rp('a', 2))
    assert h.push(rp('c', 0.5))
    assert not h.push(rp('d', 0.6))
    assert 0.1 == h.best
    assert 2 == len(h)
    assert h.full
    assert 'a' not in h
    assert 'd' not in h
    assert 'b' in h
    assert 'c' in h
    assert [1.0, 0.1, 0.1, 0.1, 0.1] == h.bests
    assert h.push(rp('e', 0.01))
    assert [1.0, 0.1, 0.1, 0.1, 0.1, 0.01] == h.bests
    assert 'b' in h and 'e' in h and (2 == len(h))
    assert h.push(rp('e', 5))
    assert [1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01] == h.bests
    values = {x.trial_id: x for x in h.values()}
    assert 'b' in values and 'e' in values and (2 == len(values))
    assert 5.0 == values['e'].sort_metric