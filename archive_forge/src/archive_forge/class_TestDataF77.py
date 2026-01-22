import os
import pytest
import numpy as np
from . import util
from numpy.f2py.crackfortran import crackfortran
class TestDataF77(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'data_common.f')]

    def test_data_stmts(self):
        assert self.module.mycom.mydata == 0

    def test_crackedlines(self):
        mod = crackfortran(str(self.sources[0]))
        print(mod[0]['vars'])
        assert mod[0]['vars']['mydata']['='] == '0'