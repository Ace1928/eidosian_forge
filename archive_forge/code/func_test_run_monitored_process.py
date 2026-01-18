from time import sleep
from pytest import raises
from triad.utils.convert import to_timedelta
from tune._utils import run_monitored_process
from tune.exceptions import TuneInterrupted
def test_run_monitored_process():
    assert 10 == run_monitored_process(t1, [1], {}, lambda: True, '5sec')
    with raises(TuneInterrupted):
        run_monitored_process(t1, [1], dict(wait='20sec'), lambda: True, '0.2sec')
    assert 10 == run_monitored_process(t1, [1], dict(wait='1sec'), lambda: False, '0.2sec')
    with raises(NotImplementedError):
        run_monitored_process(t2, [], {}, lambda: True, '5sec')
    assert run_monitored_process(t3, [], {}, lambda: True, '5sec') is None