from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
def _rule_recursive_instantiator_from_type(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    """The bulkiest bit: recursively analyze the type annotation and use it to determine
    how to instantiate it given some string from the commandline.

    Important: as far as argparse is concerned, all inputs are strings.

    Conversions from strings to our desired types happen in the instantiator; this is a
    bit more flexible, and lets us handle more complex types like enums and multi-type
    tuples."""
    if _markers.Fixed in arg.field.markers:
        return dataclasses.replace(lowered, instantiator=None, metavar='{fixed}', required=False, default=_fields.MISSING_PROP)
    if lowered.instantiator is not None:
        return lowered
    try:
        instantiator, metadata = _instantiators.instantiator_from_type(arg.field.type_or_callable, arg.type_from_typevar, arg.field.markers)
    except _instantiators.UnsupportedTypeAnnotationError as e:
        if arg.field.default in _fields.MISSING_SINGLETONS:
            raise _instantiators.UnsupportedTypeAnnotationError(f'Unsupported type annotation for the field {_strings.make_field_name([arg.extern_prefix, arg.field.intern_name])}. To suppress this error, assign the field a default value.') from e
        else:
            return dataclasses.replace(lowered, metavar='{fixed}', required=False, default=_fields.MISSING_PROP)
    if metadata.action == 'append':

        def append_instantiator(x: Any) -> Any:
            out = instantiator(x)
            if arg.field.default in _fields.MISSING_SINGLETONS:
                return instantiator(x)
            return type(out)(arg.field.default) + out
        return dataclasses.replace(lowered, instantiator=append_instantiator, default=None, choices=metadata.choices, nargs=metadata.nargs, metavar=metadata.metavar, action=metadata.action, required=False)
    else:
        return dataclasses.replace(lowered, instantiator=instantiator, choices=metadata.choices, nargs=metadata.nargs, metavar=metadata.metavar, action=metadata.action)