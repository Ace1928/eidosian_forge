import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _create_param_dict(self, func_args):
    for i, a in enumerate(func_args):
        if isinstance(a, IndexedBase):
            self.param_dict[a] = (self.fn.args[i], i)
            self.fn.args[i].name = str(a)
        else:
            self.param_dict[a] = (self.fn.args[self.signature.input_arg], i)