import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
def test_notPublicPrivate(self, tmp_path):
    fpath = util.getpath('tests', 'src', 'crackfortran', 'pubprivmod.f90')
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    mod = mod[0]
    assert mod['vars']['a']['attrspec'] == ['private']
    assert mod['vars']['b']['attrspec'] == ['public']
    assert mod['vars']['seta']['attrspec'] == ['public']