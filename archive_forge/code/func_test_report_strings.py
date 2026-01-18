import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def test_report_strings():
    rep = Report()
    assert rep.__str__() != ''
    assert rep.message == ''
    str_io = StringIO()
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    rep = Report(ValueError, 20, 'msg', 'fix')
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    rep.problem_level = 30
    rep.write_raise(str_io)
    assert str_io.getvalue() == 'Level 30: msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)
    rep.fix_msg = ''
    rep.write_raise(str_io)
    assert str_io.getvalue() == 'Level 30: msg\n'
    rep.fix_msg = 'fix'
    str_io.truncate(0)
    str_io.seek(0)
    rep.problem_level = 20
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    rep.write_raise(str_io, log_level=20)
    assert str_io.getvalue() == 'Level 20: msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)
    with pytest.raises(ValueError):
        rep.write_raise(str_io, 20)
    assert str_io.getvalue() == ''
    with pytest.raises(ValueError):
        rep.write_raise(str_io, 20, 20)
    assert str_io.getvalue() == 'Level 20: msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)
    rep.error = None
    rep.write_raise(str_io, 20)
    assert str_io.getvalue() == ''