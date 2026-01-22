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
def _instantiator_from_sequence(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Instantiator for variable-length sequences: list, sets, Tuple[T, ...], etc."""
    container_type = get_origin(typ)
    assert container_type is not None
    if container_type is collections.abc.Sequence:
        container_type = list
    if container_type is tuple:
        contained_type, ell = get_args(typ)
        assert ell == Ellipsis
    else:
        contained_type, = get_args(typ)
    if _markers.UseAppendAction in markers:
        make, inner_meta = _instantiator_from_type_inner(contained_type, type_from_typevar, allow_sequences=True, markers=markers - {_markers.UseAppendAction})

        def append_sequence_instantiator(strings: List[List[str]]) -> Any:
            assert strings is not None
            return container_type((cast(_StandardInstantiator, make)(s) for s in strings))
        return (append_sequence_instantiator, InstantiatorMetadata(nargs=inner_meta.nargs, metavar=inner_meta.metavar, choices=inner_meta.choices, action='append'))
    else:
        make, inner_meta = _instantiator_from_type_inner(contained_type, type_from_typevar, allow_sequences='fixed_length', markers=markers)

        def sequence_instantiator(strings: List[str]) -> Any:
            if isinstance(inner_meta.nargs, int) and len(strings) % inner_meta.nargs != 0:
                raise ValueError(f'input {strings} is of length {len(strings)}, which is not divisible by {inner_meta.nargs}.')
            out = []
            step = inner_meta.nargs if isinstance(inner_meta.nargs, int) else 1
            for i in range(0, len(strings), step):
                out.append(make(strings[i:i + inner_meta.nargs]))
            assert container_type is not None
            return container_type(out)
        return (sequence_instantiator, InstantiatorMetadata(nargs='*', metavar=_strings.multi_metavar_from_single(inner_meta.metavar), choices=inner_meta.choices, action=None))