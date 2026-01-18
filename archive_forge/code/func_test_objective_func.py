from fs.base import FS as FSBase
from triad import FileSystem
from tune.iterative.objective import (
from tune.concepts.flow import Trial, TrialDecision, TrialJudge, TrialReport, Monitor
def test_objective_func(tmpdir):
    fs = FileSystem().opendir(str(tmpdir))
    j = J([3, 3, 2])
    f = F().copy()
    t = Trial('abc', {'a': 1})
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    assert 6.0 == j.report.metric
    assert -6.0 == j.report.sort_metric
    f.run(t, judge=j, checkpoint_basedir_fs=fs)
    assert -10 == f.v
    assert 8.0 == j.report.metric
    assert -8.0 == j.report.sort_metric