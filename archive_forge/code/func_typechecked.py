from __future__ import annotations
import ast
import inspect
import sys
from collections.abc import Sequence
from functools import partial
from inspect import isclass, isfunction
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, TypeVar, cast, overload
from warnings import warn
from ._config import CollectionCheckStrategy, ForwardRefPolicy, global_config
from ._exceptions import InstrumentationWarning
from ._functions import TypeCheckFailCallback
from ._transformer import TypeguardTransformer
from ._utils import Unset, function_name, get_stacklevel, is_method_of, unset
def typechecked(target: T_CallableOrType | None=None, *, forward_ref_policy: ForwardRefPolicy | Unset=unset, typecheck_fail_callback: TypeCheckFailCallback | Unset=unset, collection_check_strategy: CollectionCheckStrategy | Unset=unset, debug_instrumentation: bool | Unset=unset) -> Any:
    """
    Instrument the target function to perform run-time type checking.

    This decorator recompiles the target function, injecting code to type check
    arguments, return values, yield values (excluding ``yield from``) and assignments to
    annotated local variables.

    This can also be used as a class decorator. This will instrument all type annotated
    methods, including :func:`@classmethod <classmethod>`,
    :func:`@staticmethod <staticmethod>`,  and :class:`@property <property>` decorated
    methods in the class.

    .. note:: When Python is run in optimized mode (``-O`` or ``-OO``, this decorator
        is a no-op). This is a feature meant for selectively introducing type checking
        into a code base where the checks aren't meant to be run in production.

    :param target: the function or class to enable type checking for
    :param forward_ref_policy: override for
        :attr:`.TypeCheckConfiguration.forward_ref_policy`
    :param typecheck_fail_callback: override for
        :attr:`.TypeCheckConfiguration.typecheck_fail_callback`
    :param collection_check_strategy: override for
        :attr:`.TypeCheckConfiguration.collection_check_strategy`
    :param debug_instrumentation: override for
        :attr:`.TypeCheckConfiguration.debug_instrumentation`

    """
    if target is None:
        return partial(typechecked, forward_ref_policy=forward_ref_policy, typecheck_fail_callback=typecheck_fail_callback, collection_check_strategy=collection_check_strategy, debug_instrumentation=debug_instrumentation)
    if not __debug__:
        return target
    if isclass(target):
        for key, attr in target.__dict__.items():
            if is_method_of(attr, target):
                retval = instrument(attr)
                if isfunction(retval):
                    setattr(target, key, retval)
            elif isinstance(attr, (classmethod, staticmethod)):
                if is_method_of(attr.__func__, target):
                    retval = instrument(attr.__func__)
                    if isfunction(retval):
                        wrapper = attr.__class__(retval)
                        setattr(target, key, wrapper)
            elif isinstance(attr, property):
                kwargs: dict[str, Any] = dict(doc=attr.__doc__)
                for name in ('fset', 'fget', 'fdel'):
                    property_func = kwargs[name] = getattr(attr, name)
                    if is_method_of(property_func, target):
                        retval = instrument(property_func)
                        if isfunction(retval):
                            kwargs[name] = retval
                setattr(target, key, attr.__class__(**kwargs))
        return target
    wrapper_class: type[classmethod[Any, Any, Any]] | type[staticmethod[Any, Any]] | None = None
    if isinstance(target, (classmethod, staticmethod)):
        wrapper_class = target.__class__
        target = target.__func__
    retval = instrument(target)
    if isinstance(retval, str):
        warn(f'{retval} -- not typechecking {function_name(target)}', InstrumentationWarning, stacklevel=get_stacklevel())
        return target
    if wrapper_class is None:
        return retval
    else:
        return wrapper_class(retval)