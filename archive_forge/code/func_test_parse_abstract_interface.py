from pathlib import Path
import pytest
import textwrap
from . import util
from numpy.f2py import crackfortran
from numpy.testing import IS_WASM
def test_parse_abstract_interface(self):
    fpath = util.getpath('tests', 'src', 'abstract_interface', 'gh18403_mod.f90')
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    assert len(mod[0]['body']) == 1
    assert mod[0]['body'][0]['block'] == 'abstract interface'