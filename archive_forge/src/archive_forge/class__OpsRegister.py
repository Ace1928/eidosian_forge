import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
class _OpsRegister:
    """
    Holds the ops for each dtypes signature like ('ff->f', func1)
    and allows to do look ups for these
    """

    class _Op:

        def __init__(self, in_types, out_types, func):
            self.func = func
            self.in_types = tuple((numpy.dtype(i) for i in in_types))
            self.out_types = tuple((numpy.dtype(o) for o in out_types))
            self.sig_str = ''.join((in_t.char for in_t in self.in_types)) + '->' + ''.join((out_t.char for out_t in self.out_types))

    def __init__(self, signatures, default_func, nin, nout, name):
        self._default_func = default_func
        self._nin = nin
        self._nout = nout
        self._ops = self._process_signatures(signatures)
        self._name = name

    def _sig_str_to_tuple(self, sig):
        sig = sig.replace(' ', '')
        toks = sig.split('->')
        if len(toks) != 2:
            raise ValueError(f'signature {sig} for dtypes is invalid')
        else:
            ins, outs = toks
        return (ins, outs)

    def _process_signatures(self, signatures):
        ops = []
        for sig in signatures:
            if isinstance(sig, tuple):
                sig, op = sig
            else:
                op = self._default_func
            ins, outs = self._sig_str_to_tuple(sig)
            if len(ins) != self._nin:
                raise ValueError(f'signature {sig} for dtypes is invalid number of inputs is not consistent with general signature')
            if len(outs) != self._nout:
                raise ValueError(f'signature {sig} for dtypes is invalid number of inputs is not consistent with general signature')
            ops.append(_OpsRegister._Op(ins, outs, op))
        return ops

    def _determine_from_args(self, args, casting):
        n = len(args)
        in_types = tuple((arg.dtype for arg in args))
        for op in self._ops:
            op_types = op.in_types
            for i in range(n):
                it = in_types[i]
                ot = op_types[i]
                if not numpy.can_cast(it, ot, casting=casting):
                    break
            else:
                return op
        return None

    def _determine_from_dtype(self, dtype):
        for op in self._ops:
            op_types = op.out_types
            for t in op_types:
                if t != dtype:
                    break
            else:
                return op
        return None

    def _determine_from_signature(self, signature):
        if isinstance(signature, tuple):
            if len(signature) == 1:
                raise TypeError('The use of a length 1 tuple for the ufunc `signature` is not allowed. Use `dtype` or  fill the tuple with `None`s.')
            nin = self._nin
            nout = self._nout
            if len(signature) != nin + nout:
                raise TypeError(f'A type-tuple must be specified of length 1 or 3 for ufunc {self._name}')
            signature = ''.join((numpy.dtype(t).char for t in signature[:nin])) + '->' + ''.join((numpy.dtype(t).char for t in signature[nin:nin + nout]))
        if isinstance(signature, str):
            is_out = len(signature) == 1
            for op in self._ops:
                if is_out:
                    for t in op.out_types:
                        if t.char != signature:
                            break
                    else:
                        return op
                elif op.sig_str == signature:
                    return op
        raise TypeError(f'No loop matching the specified signature and casting was found for ufunc {self._name}')

    def determine_dtype(self, args, dtype, casting, signature):
        ret_dtype = None
        func = self._default_func
        if signature is not None:
            op = self._determine_from_signature(signature)
        elif dtype is not None:
            if type(dtype) == tuple:
                raise RuntimeError('dtype with tuple is not yet supported')
            op = self._determine_from_dtype(dtype)
        else:
            op = self._determine_from_args(args, casting)
        if op is None:
            if dtype is None:
                dtype = args[0].dtype
                for arg in args:
                    ret_dtype = numpy.promote_types(dtype, arg.dtype)
            else:
                ret_dtype = get_dtype(dtype)
        else:
            n_args = []

            def argname():
                return f'ufunc {self._name} input {i}'
            for i, (arg, in_type) in enumerate(zip(args, op.in_types)):
                _raise_if_invalid_cast(arg.dtype, in_type, casting, argname)
                n_args.append(arg.astype(in_type, copy=False))
            args = n_args
            ret_dtype = op.out_types[0]
            func = op.func
        return (args, ret_dtype, func)