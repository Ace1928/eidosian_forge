import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _process_class(cls, init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots, weakref_slot):
    fields = {}
    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:
        globals = {}
    setattr(cls, _PARAMS, _DataclassParams(init, repr, eq, order, unsafe_hash, frozen))
    any_frozen_base = False
    has_dataclass_bases = False
    for b in cls.__mro__[-1:0:-1]:
        base_fields = getattr(b, _FIELDS, None)
        if base_fields is not None:
            has_dataclass_bases = True
            for f in base_fields.values():
                fields[f.name] = f
            if getattr(b, _PARAMS).frozen:
                any_frozen_base = True
    cls_annotations = cls.__dict__.get('__annotations__', {})
    cls_fields = []
    KW_ONLY_seen = False
    dataclasses = sys.modules[__name__]
    for name, type in cls_annotations.items():
        if _is_kw_only(type, dataclasses) or (isinstance(type, str) and _is_type(type, cls, dataclasses, dataclasses.KW_ONLY, _is_kw_only)):
            if KW_ONLY_seen:
                raise TypeError(f'{name!r} is KW_ONLY, but KW_ONLY has already been specified')
            KW_ONLY_seen = True
            kw_only = True
        else:
            cls_fields.append(_get_field(cls, name, type, kw_only))
    for f in cls_fields:
        fields[f.name] = f
        if isinstance(getattr(cls, f.name, None), Field):
            if f.default is MISSING:
                delattr(cls, f.name)
            else:
                setattr(cls, f.name, f.default)
    for name, value in cls.__dict__.items():
        if isinstance(value, Field) and (not name in cls_annotations):
            raise TypeError(f'{name!r} is a field but has no type annotation')
    if has_dataclass_bases:
        if any_frozen_base and (not frozen):
            raise TypeError('cannot inherit non-frozen dataclass from a frozen one')
        if not any_frozen_base and frozen:
            raise TypeError('cannot inherit frozen dataclass from a non-frozen one')
    setattr(cls, _FIELDS, fields)
    class_hash = cls.__dict__.get('__hash__', MISSING)
    has_explicit_hash = not (class_hash is MISSING or (class_hash is None and '__eq__' in cls.__dict__))
    if order and (not eq):
        raise ValueError('eq must be true if order is true')
    all_init_fields = [f for f in fields.values() if f._field_type in (_FIELD, _FIELD_INITVAR)]
    std_init_fields, kw_only_init_fields = _fields_in_init_order(all_init_fields)
    if init:
        has_post_init = hasattr(cls, _POST_INIT_NAME)
        _set_new_attribute(cls, '__init__', _init_fn(all_init_fields, std_init_fields, kw_only_init_fields, frozen, has_post_init, '__dataclass_self__' if 'self' in fields else 'self', globals, slots))
    field_list = [f for f in fields.values() if f._field_type is _FIELD]
    if repr:
        flds = [f for f in field_list if f.repr]
        _set_new_attribute(cls, '__repr__', _repr_fn(flds, globals))
    if eq:
        flds = [f for f in field_list if f.compare]
        self_tuple = _tuple_str('self', flds)
        other_tuple = _tuple_str('other', flds)
        _set_new_attribute(cls, '__eq__', _cmp_fn('__eq__', '==', self_tuple, other_tuple, globals=globals))
    if order:
        flds = [f for f in field_list if f.compare]
        self_tuple = _tuple_str('self', flds)
        other_tuple = _tuple_str('other', flds)
        for name, op in [('__lt__', '<'), ('__le__', '<='), ('__gt__', '>'), ('__ge__', '>=')]:
            if _set_new_attribute(cls, name, _cmp_fn(name, op, self_tuple, other_tuple, globals=globals)):
                raise TypeError(f'Cannot overwrite attribute {name} in class {cls.__name__}. Consider using functools.total_ordering')
    if frozen:
        for fn in _frozen_get_del_attr(cls, field_list, globals):
            if _set_new_attribute(cls, fn.__name__, fn):
                raise TypeError(f'Cannot overwrite attribute {fn.__name__} in class {cls.__name__}')
    hash_action = _hash_action[bool(unsafe_hash), bool(eq), bool(frozen), has_explicit_hash]
    if hash_action:
        cls.__hash__ = hash_action(cls, field_list, globals)
    if not getattr(cls, '__doc__'):
        try:
            text_sig = str(inspect.signature(cls)).replace(' -> None', '')
        except (TypeError, ValueError):
            text_sig = ''
        cls.__doc__ = cls.__name__ + text_sig
    if match_args:
        _set_new_attribute(cls, '__match_args__', tuple((f.name for f in std_init_fields)))
    if weakref_slot and (not slots):
        raise TypeError('weakref_slot is True but slots is False')
    if slots:
        cls = _add_slots(cls, frozen, weakref_slot)
    abc.update_abstractmethods(cls)
    return cls