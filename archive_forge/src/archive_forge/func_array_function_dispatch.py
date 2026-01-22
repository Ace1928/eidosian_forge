import collections
import functools
import os
from .._utils import set_module
from .._utils._inspect import getargspec
from numpy.core._multiarray_umath import (
def array_function_dispatch(dispatcher=None, module=None, verify=True, docs_from_dispatcher=False):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable or None
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__array_function__``.

        If `None`, the first argument is used as the single `like=` argument
        and not passed on.  A function implementing `like=` must call its
        dispatcher with `like` as the first non-keyword argument.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.

    """

    def decorator(implementation):
        if verify:
            if dispatcher is not None:
                verify_matching_signatures(implementation, dispatcher)
            else:
                co = implementation.__code__
                last_arg = co.co_argcount + co.co_kwonlyargcount - 1
                last_arg = co.co_varnames[last_arg]
                if last_arg != 'like' or co.co_kwonlyargcount == 0:
                    raise RuntimeError(f'__array_function__ expects `like=` to be the last argument and a keyword-only argument. {implementation} does not seem to comply.')
        if docs_from_dispatcher:
            add_docstring(implementation, dispatcher.__doc__)
        public_api = _ArrayFunctionDispatcher(dispatcher, implementation)
        public_api = functools.wraps(implementation)(public_api)
        if module is not None:
            public_api.__module__ = module
        ARRAY_FUNCTIONS.add(public_api)
        return public_api
    return decorator