import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def _instantiator_from_dict(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    key_type, val_type = get_args(typ)
    key_instantiator, key_meta = _instantiator_from_type_inner(key_type, type_from_typevar, allow_sequences='fixed_length', markers=markers)
    if _markers.UseAppendAction in markers:
        val_instantiator, val_meta = _instantiator_from_type_inner(val_type, type_from_typevar, allow_sequences=True, markers=markers - {_markers.UseAppendAction})
        pair_metavar = f'{key_meta.metavar} {val_meta.metavar}'
        key_nargs = cast(int, key_meta.nargs)
        val_nargs = val_meta.nargs
        assert isinstance(key_nargs, int)

        def append_dict_instantiator(strings: List[List[str]]) -> Any:
            out = {}
            for s in strings:
                out[key_instantiator(s[:key_nargs])] = val_instantiator(s[key_nargs:])
            return out
        return (append_dict_instantiator, InstantiatorMetadata(nargs=key_nargs + val_nargs if isinstance(val_nargs, int) else '*', metavar=pair_metavar, choices=None, action='append'))
    else:
        val_instantiator, val_meta = _instantiator_from_type_inner(val_type, type_from_typevar, allow_sequences='fixed_length', markers=markers)
        pair_metavar = f'{key_meta.metavar} {val_meta.metavar}'
        key_nargs = cast(int, key_meta.nargs)
        val_nargs = cast(int, val_meta.nargs)
        assert isinstance(key_nargs, int)
        assert isinstance(val_nargs, int)
        pair_nargs = key_nargs + val_nargs

        def dict_instantiator(strings: List[str]) -> Any:
            out = {}
            if len(strings) % pair_nargs != 0:
                raise ValueError('incomplete set of key value pairs!')
            index = 0
            for _ in range(len(strings) // pair_nargs):
                assert isinstance(key_nargs, int)
                assert isinstance(val_nargs, int)
                k = strings[index:index + key_nargs]
                index += key_nargs
                v = strings[index:index + val_nargs]
                index += val_nargs
                if key_meta.choices is not None and any((kj not in key_meta.choices for kj in k)):
                    raise ValueError(f'invalid choice: {k} (choose from {key_meta.choices}))')
                if val_meta.choices is not None and any((vj not in val_meta.choices for vj in v)):
                    raise ValueError(f'invalid choice: {v} (choose from {val_meta.choices}))')
                out[key_instantiator(k)] = val_instantiator(v)
            return out
        return (dict_instantiator, InstantiatorMetadata(nargs='*', metavar=_strings.multi_metavar_from_single(pair_metavar), choices=None, action=None))