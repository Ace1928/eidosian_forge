import os
import pytest
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_matlab, reason='matlab is not available')
def test_run_interface(tmpdir):
    default_script_file = clean_workspace_and_get_default_script_file()
    mc = mlab.MatlabCommand(matlab_cmd='foo_m')
    assert not os.path.exists(default_script_file), 'scriptfile should not exist 1.'
    with pytest.raises(ValueError):
        mc.run()
    assert not os.path.exists(default_script_file), 'scriptfile should not exist 2.'
    if os.path.exists(default_script_file):
        os.remove(default_script_file)
    mc.inputs.script = 'a=1;'
    assert not os.path.exists(default_script_file), 'scriptfile should not exist 3.'
    with pytest.raises(IOError):
        mc.run()
    assert os.path.exists(default_script_file), 'scriptfile should exist 3.'
    if os.path.exists(default_script_file):
        os.remove(default_script_file)
    cwd = tmpdir.chdir()
    mc = mlab.MatlabCommand(script='foo;', paths=[tmpdir.strpath], mfile=True)
    assert not os.path.exists(default_script_file), 'scriptfile should not exist 4.'
    with pytest.raises(OSError):
        mc.run()
    assert os.path.exists(default_script_file), 'scriptfile should exist 4.'
    if os.path.exists(default_script_file):
        os.remove(default_script_file)
    res = mlab.MatlabCommand(script='a=1;', paths=[tmpdir.strpath], mfile=True).run()
    assert res.runtime.returncode == 0
    assert os.path.exists(default_script_file), 'scriptfile should exist 5.'
    cwd.chdir()