import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def __param_inheritance(mcs, param_name, param):
    """
        Look for Parameter values in superclasses of this
        Parameterized class.

        Ordinarily, when a Python object is instantiated, attributes
        not given values in the constructor will inherit the value
        given in the object's class, or in its superclasses.  For
        Parameters owned by Parameterized classes, we have implemented
        an additional level of default lookup, should this ordinary
        lookup return only `Undefined`.

        In such a case, i.e. when no non-`Undefined` value was found for a
        Parameter by the usual inheritance mechanisms, we explicitly
        look for Parameters with the same name in superclasses of this
        Parameterized class, and use the first such value that we
        find.

        The goal is to be able to set the default value (or other
        slots) of a Parameter within a Parameterized class, just as we
        can set values for non-Parameter objects in Parameterized
        classes, and have the values inherited through the
        Parameterized hierarchy as usual.

        Note that instantiate is handled differently: if there is a
        parameter with the same name in one of the superclasses with
        instantiate set to True, this parameter will inherit
        instantiate=True.
        """
    p_type = type(param)
    slots = dict.fromkeys(p_type._all_slots_)
    setattr(param, 'owner', mcs)
    del slots['owner']
    if 'objtype' in slots:
        setattr(param, 'objtype', mcs)
        del slots['objtype']
    supers = classlist(mcs)[::-1]
    type_change = False
    for superclass in supers:
        super_param = superclass.__dict__.get(param_name)
        if not isinstance(super_param, Parameter):
            continue
        if super_param.instantiate is True:
            param.instantiate = True
        super_type = type(super_param)
        if not issubclass(super_type, p_type):
            type_change = True
    del slots['instantiate']
    callables, slot_values = ({}, {})
    slot_overridden = False
    for slot in slots.keys():
        for scls in supers:
            new_param = scls.__dict__.get(param_name)
            if new_param is None or not hasattr(new_param, slot):
                continue
            new_value = getattr(new_param, slot)
            old_value = slot_values.get(slot, Undefined)
            if new_value is Undefined:
                continue
            elif new_value is old_value:
                continue
            elif old_value is Undefined:
                slot_values[slot] = new_value
                if slot_overridden or type_change:
                    break
            else:
                if slot not in param._non_validated_slots:
                    slot_overridden = True
                break
        if slot_values.get(slot, Undefined) is Undefined:
            try:
                default_val = param._slot_defaults[slot]
            except KeyError as e:
                raise KeyError(f'Slot {slot!r} of parameter {param_name!r} has no default value defined in `_slot_defaults`') from e
            if callable(default_val):
                callables[slot] = default_val
            else:
                slot_values[slot] = default_val
        elif slot == 'allow_refs':
            explicit_no_refs = mcs._param__private.explicit_no_refs
            if param.allow_refs is False:
                explicit_no_refs.append(param.name)
            elif param.allow_refs is True and param.name in explicit_no_refs:
                explicit_no_refs.remove(param.name)
    for slot, value in slot_values.items():
        setattr(param, slot, value)
        if slot != 'default':
            v = getattr(param, slot)
            if _is_mutable_container(v):
                setattr(param, slot, copy.copy(v))
    for slot, fn in callables.items():
        setattr(param, slot, fn(param))
    param._update_state()
    if type_change or (slot_overridden and param.default is not None):
        try:
            param._validate(param.default)
        except Exception as e:
            msg = f'{_validate_error_prefix(param)} failed to validate its default value on class creation, this is going to raise an error in the future. '
            parents = ', '.join((klass.__name__ for klass in mcs.__mro__[1:-2]))
            if not type_change and slot_overridden:
                msg += f'The Parameter is defined with attributes which when combined with attributes inherited from its parent classes ({parents}) make it invalid. Please fix the Parameter attributes.'
            elif type_change and (not slot_overridden):
                msg += f'The Parameter type changed between class {mcs.__name__!r} and one of its parent classes ({parents}) which made it invalid. Please fix the Parameter type.'
            else:
                pass
            msg += f'\nValidation failed with:\n{e}'
            warnings.warn(msg, category=_ParamFutureWarning, stacklevel=4)