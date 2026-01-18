import os
import pytest
from nipype.interfaces import r
@pytest.mark.skipif(no_r, reason='R is not available')
def test_set_rcmd(tmpdir):
    cwd = tmpdir.chdir()
    default_script_file = r.RInputSpec().script_file
    ri = r.RCommand()
    _default_r_cmd = ri._cmd
    ri.set_default_r_cmd('foo')
    assert not os.path.exists(default_script_file), 'scriptfile should not exist.'
    assert ri._cmd == 'foo'
    ri.set_default_r_cmd(_default_r_cmd)
    cwd.chdir()