import os
from shutil import rmtree
import pytest
from nipype.utils.misc import (
def test_rgetcwd(monkeypatch, tmpdir):
    from ..misc import rgetcwd
    oldpath = tmpdir.strpath
    tmpdir.mkdir('sub').chdir()
    newpath = os.getcwd()
    assert rgetcwd() == newpath
    rmtree(newpath, ignore_errors=True)
    with pytest.raises(OSError):
        os.getcwd()
    monkeypatch.setenv('PWD', oldpath)
    assert rgetcwd(error=False) == oldpath
    with pytest.raises(OSError):
        rgetcwd()
    monkeypatch.delenv('PWD')
    with pytest.raises(OSError):
        rgetcwd(error=False)