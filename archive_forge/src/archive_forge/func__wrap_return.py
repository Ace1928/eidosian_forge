import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _wrap_return(self, lj, vals):
    if self.signature.ret_type == ctypes.c_double:
        return vals[0]
    void_ptr = ll.PointerType(ll.IntType(32))
    wrap_type = ll.FunctionType(void_ptr, [self.fp_type])
    wrap_fn = ll.Function(lj.module, wrap_type, 'PyFloat_FromDouble')
    wrapped_vals = [lj.builder.call(wrap_fn, [v]) for v in vals]
    if len(vals) == 1:
        final_val = wrapped_vals[0]
    else:
        tuple_arg_types = [ll.IntType(32)]
        tuple_arg_types.extend([void_ptr] * len(vals))
        tuple_type = ll.FunctionType(void_ptr, tuple_arg_types)
        tuple_fn = ll.Function(lj.module, tuple_type, 'PyTuple_Pack')
        tuple_args = [ll.Constant(ll.IntType(32), len(wrapped_vals))]
        tuple_args.extend(wrapped_vals)
        final_val = lj.builder.call(tuple_fn, tuple_args)
    return final_val