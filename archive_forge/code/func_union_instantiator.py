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