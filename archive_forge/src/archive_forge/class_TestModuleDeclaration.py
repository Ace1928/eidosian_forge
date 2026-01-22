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
class TestModuleDeclaration:

    def test_dependencies(self, tmp_path):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'foo_deps.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert mod[0]['vars']['abar']['='] == "bar('abar')"