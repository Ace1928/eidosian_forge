import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
def _test_call_functions(self):
    apiname = libname[5:]
    apifunc = getattr(libdevice, apiname)
    retty, args = functions[libname]
    sig = create_signature(retty, args)
    funcargs = ', '.join(['a%d' % i for i, arg in enumerate(args) if not arg.is_ptr])
    if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
        pyargs = ', '.join(['r%d' % i for i in range(len(sig.return_type))])
        pyargs += ', ' + funcargs
        retvars = ', '.join(['r%d[0]' % i for i in range(len(sig.return_type))])
    else:
        pyargs = 'r0, ' + funcargs
        retvars = 'r0[0]'
    d = {'func': apiname, 'pyargs': pyargs, 'funcargs': funcargs, 'retvars': retvars}
    code = function_template % d
    locals = {}
    exec(code, globals(), locals)
    pyfunc = locals['pyfunc']
    pyargs = [arg.ty for arg in args if not arg.is_ptr]
    if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
        pyreturns = [ret[::1] for ret in sig.return_type]
        pyargs = pyreturns + pyargs
    else:
        pyargs.insert(0, sig.return_type[::1])
    pyargs = tuple(pyargs)
    ptx, resty = compile_ptx(pyfunc, pyargs)
    self.assertIn('ld.param', ptx)
    self.assertIn('st.global', ptx)