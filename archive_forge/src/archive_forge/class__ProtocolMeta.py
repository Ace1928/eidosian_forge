import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class _ProtocolMeta(type(typing.Protocol)):

    def __new__(mcls, name, bases, namespace, **kwargs):
        if name == 'Protocol' and len(bases) < 2:
            pass
        elif {Protocol, typing.Protocol} & set(bases):
            for base in bases:
                if not (base in {object, typing.Generic, Protocol, typing.Protocol} or base.__name__ in _PROTO_ALLOWLIST.get(base.__module__, []) or is_protocol(base)):
                    raise TypeError(f'Protocols can only inherit from other protocols, got {base!r}')
        return abc.ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

    def __init__(cls, *args, **kwargs):
        abc.ABCMeta.__init__(cls, *args, **kwargs)
        if getattr(cls, '_is_protocol', False):
            cls.__protocol_attrs__ = _get_protocol_attrs(cls)

    def __subclasscheck__(cls, other):
        if cls is Protocol:
            return type.__subclasscheck__(cls, other)
        if getattr(cls, '_is_protocol', False) and (not _allow_reckless_class_checks()):
            if not getattr(cls, '_is_runtime_protocol', False):
                _type_check_issubclass_arg_1(other)
                raise TypeError('Instance and class checks can only be used with @runtime_checkable protocols')
            if cls.__non_callable_proto_members__ and cls.__dict__.get('__subclasshook__') is _proto_hook:
                _type_check_issubclass_arg_1(other)
                non_method_attrs = sorted(cls.__non_callable_proto_members__)
                raise TypeError(f"Protocols with non-method members don't support issubclass(). Non-method members: {str(non_method_attrs)[1:-1]}.")
        return abc.ABCMeta.__subclasscheck__(cls, other)

    def __instancecheck__(cls, instance):
        if cls is Protocol:
            return type.__instancecheck__(cls, instance)
        if not getattr(cls, '_is_protocol', False):
            return abc.ABCMeta.__instancecheck__(cls, instance)
        if not getattr(cls, '_is_runtime_protocol', False) and (not _allow_reckless_class_checks()):
            raise TypeError('Instance and class checks can only be used with @runtime_checkable protocols')
        if abc.ABCMeta.__instancecheck__(cls, instance):
            return True
        for attr in cls.__protocol_attrs__:
            try:
                val = inspect.getattr_static(instance, attr)
            except AttributeError:
                break
            if val is None and attr not in cls.__non_callable_proto_members__:
                break
        else:
            return True
        return False

    def __eq__(cls, other):
        if abc.ABCMeta.__eq__(cls, other) is True:
            return True
        return cls is Protocol and other is typing.Protocol

    def __hash__(cls) -> int:
        return type.__hash__(cls)