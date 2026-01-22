import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
from numba.core import (cgutils, config, debuginfo, itanium_mangler, types,
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv, MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel
from .cudadrv import nvvm
from numba.cuda import codegen, nvvmutils, ufuncs
from numba.cuda.models import cuda_data_manager
class CUDACABICallConv(BaseCallConv):
    """
    Calling convention aimed at matching the CUDA C/C++ ABI. The implemented
    function signature is:

        <Python return type> (<Python arguments>)

    Exceptions are unsupported in this convention.
    """

    def _make_call_helper(self, builder):
        return None

    def return_value(self, builder, retval):
        return builder.ret(retval)

    def return_user_exc(self, builder, exc, exc_args=None, loc=None, func_name=None):
        msg = 'Python exceptions are unsupported in the CUDA C/C++ ABI'
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder, status):
        msg = 'Return status is unsupported in the CUDA C/C++ ABI'
        raise NotImplementedError(msg)

    def get_function_type(self, restype, argtypes):
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        fnty = ir.FunctionType(self.get_return_type(restype), argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn), ['arg.' + a for a in args])

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        arginfo = self._get_arg_packer(argtys)
        realargs = arginfo.as_arguments(builder, args)
        code = builder.call(callee, realargs)
        status = None
        out = self.context.get_returned_value(builder, resty, code)
        return (status, out)

    def get_return_type(self, ty):
        return self.context.data_model_manager[ty].get_return_type()