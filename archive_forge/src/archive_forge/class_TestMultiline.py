import platform
import pytest
import numpy as np
from . import util
@pytest.mark.skipif(platform.system() == 'Darwin', reason='Prone to error when run with numpy/f2py/tests on mac os, but not when run in isolation')
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='32-bit builds are buggy')
class TestMultiline(util.F2PyTest):
    suffix = '.pyf'
    module_name = 'multiline'
    code = f"\npython module {module_name}\n    usercode '''\nvoid foo(int* x) {{\n    char dummy = ';';\n    *x = 42;\n}}\n'''\n    interface\n        subroutine foo(x)\n            intent(c) foo\n            integer intent(out) :: x\n        end subroutine foo\n    end interface\nend python module {module_name}\n    "

    def test_multiline(self):
        assert self.module.foo() == 42