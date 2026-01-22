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
def _instantiator_from_union(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    options = list(get_args(typ))
    if NoneType in options:
        options.remove(NoneType)
        options.insert(0, NoneType)
    instantiators = []
    metas = []
    nargs: Optional[Union[int, Literal['*']]] = 1
    first = True
    for t in options:
        a, b = _instantiator_from_type_inner(t, type_from_typevar, allow_sequences=True, markers=markers)
        instantiators.append(a)
        metas.append(b)
        if t is not NoneType:
            if first:
                nargs = b.nargs
                first = False
            elif nargs != b.nargs:
                nargs = '*'
    metavar: str
    metavar = _join_union_metavars(map(lambda x: cast(str, x.metavar), metas))

    def union_instantiator(strings: List[str]) -> Any:
        metadata: InstantiatorMetadata
        errors = []
        for i, (instantiator, metadata) in enumerate(zip(instantiators, metas)):
            if metadata.choices is not None and any((x not in metadata.choices for x in strings)):
                errors.append(f'{options[i]}: {strings} does not match choices {metadata.choices}')
                continue
            if len(strings) == metadata.nargs or metadata.nargs == '*':
                try:
                    return instantiator(strings)
                except ValueError as e:
                    errors.append(f'{options[i]}: {e.args[0]}')
            else:
                errors.append(f'{options[i]}: input length {len(strings)} did not match expected argument count {metadata.nargs}')
        raise ValueError(f'no type in {options} could be instantiated from {strings}.\n\nGot errors:  \n- ' + '\n- '.join(errors))
    return (union_instantiator, InstantiatorMetadata(nargs=nargs, metavar=metavar, choices=None, action=None))