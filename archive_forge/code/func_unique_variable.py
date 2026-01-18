from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.alignment import deep_align
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import (
from xarray.core.utils import Frozen, compat_dict_union, dict_equiv, equivalent
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def unique_variable(name: Hashable, variables: list[Variable], compat: CompatOptions='broadcast_equals', equals: bool | None=None) -> Variable:
    """Return the unique variable from a list of variables or raise MergeError.

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        Type of equality check to use.
    equals : None or bool, optional
        corresponding to result of compat test

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
    """
    out = variables[0]
    if len(variables) == 1 or compat == 'override':
        return out
    combine_method = None
    if compat == 'minimal':
        compat = 'broadcast_equals'
    if compat == 'broadcast_equals':
        dim_lengths = broadcast_dimension_size(variables)
        out = out.set_dims(dim_lengths)
    if compat == 'no_conflicts':
        combine_method = 'fillna'
    if equals is None:
        for var in variables[1:]:
            equals = getattr(out, compat)(var, equiv=lazy_array_equiv)
            if equals is not True:
                break
        if equals is None:
            out = out.compute()
            for var in variables[1:]:
                equals = getattr(out, compat)(var)
                if not equals:
                    break
    if not equals:
        raise MergeError(f"conflicting values for variable {name!r} on objects to be combined. You can skip this check by specifying compat='override'.")
    if combine_method:
        for var in variables[1:]:
            out = getattr(out, combine_method)(var)
    return out