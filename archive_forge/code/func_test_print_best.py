from tune import Trial, TrialReport
from tune_notebook import (
from tune_notebook.monitors import _ReportBin
def test_print_best():
    t1 = Trial('a', dict(a=1, b=2), keys=['x', 'y'])
    r1 = TrialReport(t1, 0.8, sort_metric=-0.8)
    t2 = Trial('b', dict(a=11, b=12), keys=['xx', 'y'])
    r2 = TrialReport(t2, 0.7, sort_metric=-0.7)
    t3 = Trial('c', dict(a=10, b=20), keys=['x', 'y'])
    r3 = TrialReport(t3, 0.9, sort_metric=-0.9)
    b = PrintBest()
    b.on_report(r3)
    b.on_report(r2)
    b.on_report(r1)