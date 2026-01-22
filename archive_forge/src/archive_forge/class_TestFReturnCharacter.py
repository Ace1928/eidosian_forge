import pytest
from numpy import array
from . import util
import platform
class TestFReturnCharacter(TestReturnCharacter):
    sources = [util.getpath('tests', 'src', 'return_character', 'foo77.f'), util.getpath('tests', 'src', 'return_character', 'foo90.f90')]

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize('name', 't0,t1,t5,s0,s1,s5,ss'.split(','))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize('name', 't0,t1,t5,ts,s0,s1,s5,ss'.split(','))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_char, name), name)