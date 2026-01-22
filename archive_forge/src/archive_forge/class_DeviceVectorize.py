from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
class DeviceVectorize(_BaseUFuncBuilder):

    def __init__(self, func, identity=None, cache=False, targetoptions={}):
        if cache:
            raise TypeError('caching is not supported')
        for opt in targetoptions:
            if opt == 'nopython':
                warnings.warn('nopython kwarg for cuda target is redundant', RuntimeWarning)
            else:
                fmt = 'Unrecognized options. '
                fmt += "cuda vectorize target does not support option: '%s'"
                raise KeyError(fmt % opt)
        self.py_func = func
        self.identity = parse_identity(identity)
        self.kernelmap = OrderedDict()

    @property
    def pyfunc(self):
        return self.py_func

    def add(self, sig=None):
        args, return_type = sigutils.normalize_signature(sig)
        devfnsig = signature(return_type, *args)
        funcname = self.pyfunc.__name__
        kernelsource = self._get_kernel_source(self._kernel_template, devfnsig, funcname)
        corefn, return_type = self._compile_core(devfnsig)
        glbl = self._get_globals(corefn)
        sig = signature(types.void, *[a[:] for a in args] + [return_type[:]])
        exec(kernelsource, glbl)
        stager = glbl['__vectorized_%s' % funcname]
        kernel = self._compile_kernel(stager, sig)
        argdtypes = tuple((to_dtype(t) for t in devfnsig.args))
        resdtype = to_dtype(return_type)
        self.kernelmap[tuple(argdtypes)] = (resdtype, kernel)

    def build_ufunc(self):
        raise NotImplementedError

    def _get_kernel_source(self, template, sig, funcname):
        args = ['a%d' % i for i in range(len(sig.args))]
        fmts = dict(name=funcname, args=', '.join(args), argitems=', '.join(('%s[__tid__]' % i for i in args)))
        return template.format(**fmts)

    def _compile_core(self, sig):
        raise NotImplementedError

    def _get_globals(self, corefn):
        raise NotImplementedError

    def _compile_kernel(self, fnobj, sig):
        raise NotImplementedError