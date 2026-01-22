import ctypes, ctypes.util, operator, sys
from . import model
class CTypesFunctionPtr(CTypesGenericPtr):
    __slots__ = ['_own_callback', '_name']
    _ctype = ctypes.CFUNCTYPE(getattr(BResult, '_ctype', None), *[BArg._ctype for BArg in BArgs], use_errno=True)
    _reftypename = BResult._get_c_name('(* &)(%s)' % (nameargs,))

    def __init__(self, init, error=None):
        import traceback
        assert not has_varargs, 'varargs not supported for callbacks'
        if getattr(BResult, '_ctype', None) is not None:
            error = BResult._from_ctypes(BResult._create_ctype_obj(error))
        else:
            error = None

        def callback(*args):
            args2 = []
            for arg, BArg in zip(args, BArgs):
                args2.append(BArg._from_ctypes(arg))
            try:
                res2 = init(*args2)
                res2 = BResult._to_ctypes(res2)
            except:
                traceback.print_exc()
                res2 = error
            if issubclass(BResult, CTypesGenericPtr):
                if res2:
                    res2 = ctypes.cast(res2, ctypes.c_void_p).value
                else:
                    res2 = None
            return res2
        if issubclass(BResult, CTypesGenericPtr):
            callback_ctype = ctypes.CFUNCTYPE(ctypes.c_void_p, *[BArg._ctype for BArg in BArgs], use_errno=True)
        else:
            callback_ctype = CTypesFunctionPtr._ctype
        self._as_ctype_ptr = callback_ctype(callback)
        self._address = ctypes.cast(self._as_ctype_ptr, ctypes.c_void_p).value
        self._own_callback = init

    @staticmethod
    def _initialize(ctypes_ptr, value):
        if value:
            raise NotImplementedError('ctypes backend: not supported: initializers for function pointers')

    def __repr__(self):
        c_name = getattr(self, '_name', None)
        if c_name:
            i = self._reftypename.index('(* &)')
            if self._reftypename[i - 1] not in ' )*':
                c_name = ' ' + c_name
            c_name = self._reftypename.replace('(* &)', c_name)
        return CTypesData.__repr__(self, c_name)

    def _get_own_repr(self):
        if getattr(self, '_own_callback', None) is not None:
            return 'calling %r' % (self._own_callback,)
        return super(CTypesFunctionPtr, self)._get_own_repr()

    def __call__(self, *args):
        if has_varargs:
            assert len(args) >= len(BArgs)
            extraargs = args[len(BArgs):]
            args = args[:len(BArgs)]
        else:
            assert len(args) == len(BArgs)
        ctypes_args = []
        for arg, BArg in zip(args, BArgs):
            ctypes_args.append(BArg._arg_to_ctypes(arg))
        if has_varargs:
            for i, arg in enumerate(extraargs):
                if arg is None:
                    ctypes_args.append(ctypes.c_void_p(0))
                    continue
                if not isinstance(arg, CTypesData):
                    raise TypeError('argument %d passed in the variadic part needs to be a cdata object (got %s)' % (1 + len(BArgs) + i, type(arg).__name__))
                ctypes_args.append(arg._arg_to_ctypes(arg))
        result = self._as_ctype_ptr(*ctypes_args)
        return BResult._from_ctypes(result)