import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def _signature_from_callable(obj, *, follow_wrapper_chains=True, skip_bound_arg=True, globals=None, locals=None, eval_str=False, sigcls):
    """Private helper function to get signature for arbitrary
    callable objects.
    """
    _get_signature_of = functools.partial(_signature_from_callable, follow_wrapper_chains=follow_wrapper_chains, skip_bound_arg=skip_bound_arg, globals=globals, locals=locals, sigcls=sigcls, eval_str=eval_str)
    if not callable(obj):
        raise TypeError('{!r} is not a callable object'.format(obj))
    if isinstance(obj, types.MethodType):
        sig = _get_signature_of(obj.__func__)
        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig
    if follow_wrapper_chains:
        obj = unwrap(obj, stop=lambda f: hasattr(f, '__signature__') or isinstance(f, types.MethodType))
        if isinstance(obj, types.MethodType):
            return _get_signature_of(obj)
    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            if not isinstance(sig, Signature):
                raise TypeError('unexpected object {!r} in __signature__ attribute'.format(sig))
            return sig
    try:
        partialmethod = obj._partialmethod
    except AttributeError:
        pass
    else:
        if isinstance(partialmethod, functools.partialmethod):
            wrapped_sig = _get_signature_of(partialmethod.func)
            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))
            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]
            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:
                return sig
            else:
                sig_params = tuple(sig.parameters.values())
                assert not sig_params or first_wrapped_param is not sig_params[0]
                new_params = (first_wrapped_param,) + sig_params
                return sig.replace(parameters=new_params)
    if isfunction(obj) or _signature_is_functionlike(obj):
        return _signature_from_function(sigcls, obj, skip_bound_arg=skip_bound_arg, globals=globals, locals=locals, eval_str=eval_str)
    if _signature_is_builtin(obj):
        return _signature_from_builtin(sigcls, obj, skip_bound_arg=skip_bound_arg)
    if isinstance(obj, functools.partial):
        wrapped_sig = _get_signature_of(obj.func)
        return _signature_get_partial(wrapped_sig, obj)
    sig = None
    if isinstance(obj, type):
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = _get_signature_of(call)
        else:
            factory_method = None
            new = _signature_get_user_defined_method(obj, '__new__')
            init = _signature_get_user_defined_method(obj, '__init__')
            for base in obj.__mro__:
                if new is not None and '__new__' in base.__dict__:
                    factory_method = new
                    break
                elif init is not None and '__init__' in base.__dict__:
                    factory_method = init
                    break
            if factory_method is not None:
                sig = _get_signature_of(factory_method)
        if sig is None:
            for base in obj.__mro__[:-1]:
                try:
                    text_sig = base.__text_signature__
                except AttributeError:
                    pass
                else:
                    if text_sig:
                        return _signature_fromstr(sigcls, base, text_sig)
            if type not in obj.__mro__:
                if obj.__init__ is object.__init__ and obj.__new__ is object.__new__:
                    return sigcls.from_callable(object)
                else:
                    raise ValueError('no signature found for builtin type {!r}'.format(obj))
    elif not isinstance(obj, _NonUserDefinedCallables):
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            try:
                sig = _get_signature_of(call)
            except ValueError as ex:
                msg = 'no signature found for {!r}'.format(obj)
                raise ValueError(msg) from ex
    if sig is not None:
        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig
    if isinstance(obj, types.BuiltinFunctionType):
        msg = 'no signature found for builtin function {!r}'.format(obj)
        raise ValueError(msg)
    raise ValueError('callable {!r} is not supported by signature'.format(obj))