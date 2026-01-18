from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def typed_kwargs(name: str, *types: KwargInfo, allow_unknown: bool=False) -> T.Callable[..., T.Any]:
    """Decorator for type checking keyword arguments.

    Used to wrap a meson DSL implementation function, where it checks various
    things about keyword arguments, including the type, and various other
    information. For non-required values it sets the value to a default, which
    means the value will always be provided.

    If type is a :class:ContainerTypeInfo, then the default value will be
    passed as an argument to the container initializer, making a shallow copy

    :param name: the name of the function, including the object it's attached to
        (if applicable)
    :param *types: KwargInfo entries for each keyword argument.
    """

    def inner(f: TV_func) -> TV_func:

        def types_description(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...]) -> str:
            candidates = []
            for t in types_tuple:
                if isinstance(t, ContainerTypeInfo):
                    candidates.append(t.description())
                else:
                    candidates.append(t.__name__)
            shouldbe = 'one of: ' if len(candidates) > 1 else ''
            shouldbe += ', '.join(candidates)
            return shouldbe

        def raw_description(t: object) -> str:
            """describe a raw type (ie, one that is not a ContainerTypeInfo)."""
            if isinstance(t, list):
                if t:
                    return f'array[{' | '.join(sorted(mesonlib.OrderedSet((type(v).__name__ for v in t))))}]'
                return 'array[]'
            elif isinstance(t, dict):
                if t:
                    return f'dict[{' | '.join(sorted(mesonlib.OrderedSet((type(v).__name__ for v in t.values()))))}]'
                return 'dict[]'
            return type(t).__name__

        def check_value_type(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...], value: T.Any) -> bool:
            for t in types_tuple:
                if isinstance(t, ContainerTypeInfo):
                    if t.check(value):
                        return True
                elif isinstance(value, t):
                    return True
            return False

        @wraps(f)
        def wrapper(*wrapped_args: T.Any, **wrapped_kwargs: T.Any) -> T.Any:

            def emit_feature_change(values: T.Dict[_T, T.Union[str, T.Tuple[str, str]]], feature: T.Union[T.Type['FeatureDeprecated'], T.Type['FeatureNew']]) -> None:
                for n, version in values.items():
                    if isinstance(version, tuple):
                        version, msg = version
                    else:
                        msg = None
                    warning: T.Optional[str] = None
                    if isinstance(n, ContainerTypeInfo):
                        if n.check_any(value):
                            warning = f'of type {n.description()}'
                    elif isinstance(n, type):
                        if isinstance(value, n):
                            warning = f'of type {n.__name__}'
                    elif isinstance(value, list):
                        if n in value:
                            warning = f'value "{n}" in list'
                    elif isinstance(value, dict):
                        if n in value.keys():
                            warning = f'value "{n}" in dict keys'
                    elif n == value:
                        warning = f'value "{n}"'
                    if warning:
                        feature.single_use(f'"{name}" keyword argument "{info.name}" {warning}', version, subproject, msg, location=node)
            node, _, _kwargs, subproject = get_callee_args(wrapped_args)
            kwargs = T.cast('T.Dict[str, object]', _kwargs)
            if not allow_unknown:
                all_names = {t.name for t in types}
                unknowns = set(kwargs).difference(all_names)
                if unknowns:
                    ustr = ', '.join([f'"{u}"' for u in sorted(unknowns)])
                    raise InvalidArguments(f'{name} got unknown keyword arguments {ustr}')
            for info in types:
                types_tuple = info.types if isinstance(info.types, tuple) else (info.types,)
                value = kwargs.get(info.name)
                if value is not None:
                    if info.since:
                        feature_name = info.name + ' arg in ' + name
                        FeatureNew.single_use(feature_name, info.since, subproject, info.since_message, location=node)
                    if info.deprecated:
                        feature_name = info.name + ' arg in ' + name
                        FeatureDeprecated.single_use(feature_name, info.deprecated, subproject, info.deprecated_message, location=node)
                    if info.listify:
                        kwargs[info.name] = value = mesonlib.listify(value)
                    if not check_value_type(types_tuple, value):
                        shouldbe = types_description(types_tuple)
                        raise InvalidArguments(f'{name} keyword argument {info.name!r} was of type {raw_description(value)} but should have been {shouldbe}')
                    if info.validator is not None:
                        msg = info.validator(value)
                        if msg is not None:
                            raise InvalidArguments(f'{name} keyword argument "{info.name}" {msg}')
                    if info.deprecated_values is not None:
                        emit_feature_change(info.deprecated_values, FeatureDeprecated)
                    if info.since_values is not None:
                        emit_feature_change(info.since_values, FeatureNew)
                elif info.required:
                    raise InvalidArguments(f'{name} is missing required keyword argument "{info.name}"')
                else:
                    assert check_value_type(types_tuple, info.default), f'In function {name} default value of {info.name} is not a valid type, got {type(info.default)} expected {types_description(types_tuple)}'
                    kwargs[info.name] = copy.copy(info.default)
                    if info.not_set_warning:
                        mlog.warning(info.not_set_warning)
                if info.convertor:
                    kwargs[info.name] = info.convertor(kwargs[info.name])
            return f(*wrapped_args, **wrapped_kwargs)
        return T.cast('TV_func', wrapper)
    return inner