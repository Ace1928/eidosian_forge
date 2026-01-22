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
def _attrs_to_init_script(attrs, frozen, slots, pre_init, pre_init_has_args, post_init, cache_hash, base_attr_map, is_exc, needs_cached_setattr, has_cls_on_setattr, attrs_init):
    """
    Return a script of an initializer for *attrs* and a dict of globals.

    The globals are expected by the generated script.

    If *frozen* is True, we cannot set the attributes directly so we use
    a cached ``object.__setattr__``.
    """
    lines = []
    if pre_init:
        lines.append('self.__attrs_pre_init__()')
    if needs_cached_setattr:
        lines.append('_setattr = _cached_setattr_get(self)')
    if frozen is True:
        if slots is True:
            fmt_setter = _setattr
            fmt_setter_with_converter = _setattr_with_converter
        else:
            lines.append('_inst_dict = self.__dict__')

            def fmt_setter(attr_name, value_var, has_on_setattr):
                if _is_slot_attr(attr_name, base_attr_map):
                    return _setattr(attr_name, value_var, has_on_setattr)
                return f"_inst_dict['{attr_name}'] = {value_var}"

            def fmt_setter_with_converter(attr_name, value_var, has_on_setattr):
                if has_on_setattr or _is_slot_attr(attr_name, base_attr_map):
                    return _setattr_with_converter(attr_name, value_var, has_on_setattr)
                return "_inst_dict['%s'] = %s(%s)" % (attr_name, _init_converter_pat % (attr_name,), value_var)
    else:
        fmt_setter = _assign
        fmt_setter_with_converter = _assign_with_converter
    args = []
    kw_only_args = []
    attrs_to_validate = []
    names_for_globals = {}
    annotations = {'return': None}
    for a in attrs:
        if a.validator:
            attrs_to_validate.append(a)
        attr_name = a.name
        has_on_setattr = a.on_setattr is not None or (a.on_setattr is not setters.NO_OP and has_cls_on_setattr)
        arg_name = a.alias
        has_factory = isinstance(a.default, Factory)
        maybe_self = 'self' if has_factory and a.default.takes_self else ''
        if a.init is False:
            if has_factory:
                init_factory_name = _init_factory_pat % (a.name,)
                if a.converter is not None:
                    lines.append(fmt_setter_with_converter(attr_name, init_factory_name + f'({maybe_self})', has_on_setattr))
                    conv_name = _init_converter_pat % (a.name,)
                    names_for_globals[conv_name] = a.converter
                else:
                    lines.append(fmt_setter(attr_name, init_factory_name + f'({maybe_self})', has_on_setattr))
                names_for_globals[init_factory_name] = a.default.factory
            elif a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, f"attr_dict['{attr_name}'].default", has_on_setattr))
                conv_name = _init_converter_pat % (a.name,)
                names_for_globals[conv_name] = a.converter
            else:
                lines.append(fmt_setter(attr_name, f"attr_dict['{attr_name}'].default", has_on_setattr))
        elif a.default is not NOTHING and (not has_factory):
            arg = f"{arg_name}=attr_dict['{attr_name}'].default"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
            if a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))
        elif has_factory:
            arg = f'{arg_name}=NOTHING'
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
            lines.append(f'if {arg_name} is not NOTHING:')
            init_factory_name = _init_factory_pat % (a.name,)
            if a.converter is not None:
                lines.append('    ' + fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                lines.append('else:')
                lines.append('    ' + fmt_setter_with_converter(attr_name, init_factory_name + '(' + maybe_self + ')', has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append('    ' + fmt_setter(attr_name, arg_name, has_on_setattr))
                lines.append('else:')
                lines.append('    ' + fmt_setter(attr_name, init_factory_name + '(' + maybe_self + ')', has_on_setattr))
            names_for_globals[init_factory_name] = a.default.factory
        else:
            if a.kw_only:
                kw_only_args.append(arg_name)
            else:
                args.append(arg_name)
            if a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))
        if a.init is True:
            if a.type is not None and a.converter is None:
                annotations[arg_name] = a.type
            elif a.converter is not None:
                t = _AnnotationExtractor(a.converter).get_first_param_type()
                if t:
                    annotations[arg_name] = t
    if attrs_to_validate:
        names_for_globals['_config'] = _config
        lines.append('if _config._run_validators is True:')
        for a in attrs_to_validate:
            val_name = '__attr_validator_' + a.name
            attr_name = '__attr_' + a.name
            lines.append(f'    {val_name}(self, {attr_name}, self.{a.name})')
            names_for_globals[val_name] = a.validator
            names_for_globals[attr_name] = a
    if post_init:
        lines.append('self.__attrs_post_init__()')
    if cache_hash:
        if frozen:
            if slots:
                init_hash_cache = "_setattr('%s', %s)"
            else:
                init_hash_cache = "_inst_dict['%s'] = %s"
        else:
            init_hash_cache = 'self.%s = %s'
        lines.append(init_hash_cache % (_hash_cache_field, 'None'))
    if is_exc:
        vals = ','.join((f'self.{a.name}' for a in attrs if a.init))
        lines.append(f'BaseException.__init__(self, {vals})')
    args = ', '.join(args)
    pre_init_args = args
    if kw_only_args:
        args += '%s*, %s' % (', ' if args else '', ', '.join(kw_only_args))
        pre_init_kw_only_args = ', '.join(['%s=%s' % (kw_arg, kw_arg) for kw_arg in kw_only_args])
        pre_init_args += ', ' if pre_init_args else ''
        pre_init_args += pre_init_kw_only_args
    if pre_init and pre_init_has_args:
        lines[0] = 'self.__attrs_pre_init__(%s)' % pre_init_args
    return ('def %s(self, %s):\n    %s\n' % ('__attrs_init__' if attrs_init else '__init__', args, '\n    '.join(lines) if lines else 'pass'), names_for_globals, annotations)