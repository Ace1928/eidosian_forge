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
class TestPublicPrivate:

    def test_defaultPrivate(self):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'privatemod.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' in mod['vars']['b']['attrspec']
        assert 'public' not in mod['vars']['b']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']

    def test_defaultPublic(self, tmp_path):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'publicmod.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']

    def test_access_type(self, tmp_path):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'accesstype.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        tt = mod[0]['vars']
        assert set(tt['a']['attrspec']) == {'private', 'bind(c)'}
        assert set(tt['b_']['attrspec']) == {'public', 'bind(c)'}
        assert set(tt['c']['attrspec']) == {'public'}

    def test_nowrap_private_proceedures(self, tmp_path):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh23879.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        pyf = crackfortran.crack2fortran(mod)
        assert 'bar' not in pyf