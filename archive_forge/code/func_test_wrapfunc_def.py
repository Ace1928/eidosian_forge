import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_wrapfunc_def(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 are included by default

    CLI :: --[no]-wrap-functions
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
    out, _ = capfd.readouterr()
    assert 'Fortran 77 wrappers are saved to' in out
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} --wrap-functions'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Fortran 77 wrappers are saved to' in out