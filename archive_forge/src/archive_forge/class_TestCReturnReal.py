import platform
import pytest
import numpy as np
from numpy import array
from . import util
@pytest.mark.skipif(platform.system() == 'Darwin', reason='Prone to error when run with numpy/f2py/tests on mac os, but not when run in isolation')
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='32-bit builds are buggy')
class TestCReturnReal(TestReturnReal):
    suffix = '.pyf'
    module_name = 'c_ext_return_real'
    code = "\npython module c_ext_return_real\nusercode '''\nfloat t4(float value) { return value; }\nvoid s4(float *t4, float value) { *t4 = value; }\ndouble t8(double value) { return value; }\nvoid s8(double *t8, double value) { *t8 = value; }\n'''\ninterface\n  function t4(value)\n    real*4 intent(c) :: t4,value\n  end\n  function t8(value)\n    real*8 intent(c) :: t8,value\n  end\n  subroutine s4(t4,value)\n    intent(c) s4\n    real*4 intent(out) :: t4\n    real*4 intent(c) :: value\n  end\n  subroutine s8(t8,value)\n    intent(c) s8\n    real*8 intent(out) :: t8\n    real*8 intent(c) :: value\n  end\nend interface\nend python module c_ext_return_real\n    "

    @pytest.mark.parametrize('name', 't4,t8,s4,s8'.split(','))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)