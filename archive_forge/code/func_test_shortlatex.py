import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_shortlatex(capfd, hello_world_f90, monkeypatch):
    """Ensures that truncated documentation is written out

    TODO: Test to ensure this has no effect without --latex-doc
    CLI :: --latex-doc --short-latex
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} --latex-doc --short-latex'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Documentation is saved to file' in out
        with Path(f'./{mname}module.tex').open() as otex:
            assert '\\documentclass' not in otex.read()