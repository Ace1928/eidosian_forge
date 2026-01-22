import functools
from .autoray import (
from . import lazy
class AutoCompiled:
    """Just in time compile a ``autoray.do`` using function. See the main
    wrapper ``autojit``.
    """

    def __init__(self, fn, backend=None, compiler_opts=None):
        self._fn = fn
        self._backend = backend
        self._compiled_fns = {}
        if compiler_opts is None:
            self._compiler_kwargs = {}
        else:
            self._compiler_kwargs = compiler_opts

    def __call__(self, *args, backend=None, **kwargs):
        array_backend = infer_backend(next(tree_iter((args, kwargs), is_array)))
        if backend is None:
            if self._backend is None:
                backend = array_backend
            else:
                backend = self._backend
        try:
            key = _backend_lookup[backend, array_backend]
        except KeyError:
            if backend in _compiler_lookup:
                key = backend
            else:
                key = f'python-{array_backend}'
            _backend_lookup[backend, array_backend] = key
        try:
            fn_compiled = self._compiled_fns[key]
        except KeyError:
            if 'python' in key:
                backend = 'python'
            backend_compiler = _compiler_lookup.get(backend, CompilePython)
            compiler_kwargs = self._compiler_kwargs.get(backend, {})
            fn_compiled = backend_compiler(self._fn, **compiler_kwargs)
            self._compiled_fns[key] = fn_compiled
        return fn_compiled(*args, array_backend=array_backend, **kwargs)