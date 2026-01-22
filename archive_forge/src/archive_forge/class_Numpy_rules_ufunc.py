import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
class Numpy_rules_ufunc(AbstractTemplate):

    @classmethod
    def _handle_inputs(cls, ufunc, args, kws):
        """
        Process argument types to a given *ufunc*.
        Returns a (base types, explicit outputs, ndims, layout) tuple where:
        - `base types` is a tuple of scalar types for each input
        - `explicit outputs` is a tuple of explicit output types (arrays)
        - `ndims` is the number of dimensions of the loop and also of
          any outputs, explicit or implicit
        - `layout` is the layout for any implicit output to be allocated
        """
        nin = ufunc.nin
        nout = ufunc.nout
        nargs = ufunc.nargs
        assert nargs == nin + nout
        if len(args) < nin:
            msg = "ufunc '{0}': not enough arguments ({1} found, {2} required)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nin))
        if len(args) > nargs:
            msg = "ufunc '{0}': too many arguments ({1} found, {2} maximum)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nargs))
        args = [a.as_array if isinstance(a, types.ArrayCompatible) else a for a in args]
        arg_ndims = [a.ndim if isinstance(a, types.ArrayCompatible) else 0 for a in args]
        ndims = max(arg_ndims)
        explicit_outputs = args[nin:]
        if not all((d == ndims for d in arg_ndims[nin:])):
            msg = "ufunc '{0}' called with unsuitable explicit output arrays."
            raise TypingError(msg=msg.format(ufunc.__name__))
        if not all((isinstance(output, types.ArrayCompatible) for output in explicit_outputs)):
            msg = "ufunc '{0}' called with an explicit output that is not an array"
            raise TypingError(msg=msg.format(ufunc.__name__))
        if not all((output.mutable for output in explicit_outputs)):
            msg = "ufunc '{0}' called with an explicit output that is read-only"
            raise TypingError(msg=msg.format(ufunc.__name__))
        base_types = [x.dtype if isinstance(x, types.ArrayCompatible) else x for x in args]
        layout = None
        if ndims > 0 and len(explicit_outputs) < ufunc.nout:
            layout = 'C'
            layouts = [x.layout if isinstance(x, types.ArrayCompatible) else '' for x in args]
            if 'C' not in layouts and 'F' in layouts:
                layout = 'F'
        return (base_types, explicit_outputs, ndims, layout)

    @property
    def ufunc(self):
        return self.key

    def generic(self, args, kws):
        args = [x.type if isinstance(x, types.Optional) else x for x in args]
        ufunc = self.ufunc
        base_types, explicit_outputs, ndims, layout = self._handle_inputs(ufunc, args, kws)
        ufunc_loop = ufunc_find_matching_loop(ufunc, base_types)
        if ufunc_loop is None:
            raise TypingError("can't resolve ufunc {0} for types {1}".format(ufunc.__name__, args))
        if not supported_ufunc_loop(ufunc, ufunc_loop):
            msg = "ufunc '{0}' using the loop '{1}' not supported in this mode"
            raise TypingError(msg=msg.format(ufunc.__name__, ufunc_loop.ufunc_sig))
        explicit_outputs_np = [as_dtype(tp.dtype) for tp in explicit_outputs]
        if not all((np.can_cast(fromty, toty, 'unsafe') for fromty, toty in zip(ufunc_loop.numpy_outputs, explicit_outputs_np))):
            msg = "ufunc '{0}' can't cast result to explicit result type"
            raise TypingError(msg=msg.format(ufunc.__name__))
        out = list(explicit_outputs)
        implicit_output_count = ufunc.nout - len(explicit_outputs)
        if implicit_output_count > 0:
            ret_tys = ufunc_loop.outputs[-implicit_output_count:]
            if ndims > 0:
                assert layout is not None
                array_ufunc_type = None
                for a in args:
                    if hasattr(a, '__array_ufunc__'):
                        array_ufunc_type = a
                        break
                output_type = types.Array
                if array_ufunc_type is not None:
                    output_type = array_ufunc_type.__array_ufunc__(ufunc, '__call__', *args, **kws)
                    if output_type is NotImplemented:
                        msg = f'unsupported use of ufunc {ufunc} on {array_ufunc_type}'
                        raise NumbaTypeError(msg)
                    elif not issubclass(output_type, types.Array):
                        msg = f'ufunc {ufunc} on {array_ufunc_type}cannot return non-array {output_type}'
                        raise TypeError(msg)
                ret_tys = [output_type(dtype=ret_ty, ndim=ndims, layout=layout) for ret_ty in ret_tys]
                ret_tys = [resolve_output_type(self.context, args, ret_ty) for ret_ty in ret_tys]
            out.extend(ret_tys)
        return _ufunc_loop_sig(out, args)