import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _transform_attrs(cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer):
    """
    Transform all `_CountingAttr`s on a class into `Attribute`s.

    If *these* is passed, use that and don't look for them on the class.

    *collect_by_mro* is True, collect them in the correct MRO order, otherwise
    use the old -- incorrect -- order.  See #428.

    Return an `_Attributes`.
    """
    cd = cls.__dict__
    anns = _get_annotations(cls)
    if these is not None:
        ca_list = list(these.items())
    elif auto_attribs is True:
        ca_names = {name for name, attr in cd.items() if isinstance(attr, _CountingAttr)}
        ca_list = []
        annot_names = set()
        for attr_name, type in anns.items():
            if _is_class_var(type):
                continue
            annot_names.add(attr_name)
            a = cd.get(attr_name, NOTHING)
            if not isinstance(a, _CountingAttr):
                a = attrib() if a is NOTHING else attrib(default=a)
            ca_list.append((attr_name, a))
        unannotated = ca_names - annot_names
        if len(unannotated) > 0:
            raise UnannotatedAttributeError('The following `attr.ib`s lack a type annotation: ' + ', '.join(sorted(unannotated, key=lambda n: cd.get(n).counter)) + '.')
    else:
        ca_list = sorted(((name, attr) for name, attr in cd.items() if isinstance(attr, _CountingAttr)), key=lambda e: e[1].counter)
    own_attrs = [Attribute.from_counting_attr(name=attr_name, ca=ca, type=anns.get(attr_name)) for attr_name, ca in ca_list]
    if collect_by_mro:
        base_attrs, base_attr_map = _collect_base_attrs(cls, {a.name for a in own_attrs})
    else:
        base_attrs, base_attr_map = _collect_base_attrs_broken(cls, {a.name for a in own_attrs})
    if kw_only:
        own_attrs = [a.evolve(kw_only=True) for a in own_attrs]
        base_attrs = [a.evolve(kw_only=True) for a in base_attrs]
    attrs = base_attrs + own_attrs
    had_default = False
    for a in (a for a in attrs if a.init is not False and a.kw_only is False):
        if had_default is True and a.default is NOTHING:
            msg = f'No mandatory attributes allowed after an attribute with a default value or factory.  Attribute in question: {a!r}'
            raise ValueError(msg)
        if had_default is False and a.default is not NOTHING:
            had_default = True
    if field_transformer is not None:
        attrs = field_transformer(cls, attrs)
    attrs = [a.evolve(alias=_default_init_alias_for(a.name)) if not a.alias else a for a in attrs]
    attr_names = [a.name for a in attrs]
    AttrsClass = _make_attr_tuple_class(cls.__name__, attr_names)
    return _Attributes((AttrsClass(attrs), base_attrs, base_attr_map))