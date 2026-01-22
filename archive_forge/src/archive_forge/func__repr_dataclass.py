import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
def _repr_dataclass(obj, *, default_values: Optional[Dict[str, Any]]=None) -> str:
    """A utility function to elegantly represent dataclasses.

    In contrast to the default dataclass `__repr__`, which shows all parameters, this
    function only shows parameters with non-default values.

    Args:
        obj: The dataclass to represent.
        default_values: An optional dictionary that maps field names to default values.
            Use this parameter to specify default values that are generated dynamically
            (e.g., in `__post_init__` or by a `default_factory`). If a default value
            isn't specified in `default_values`, then the default value is inferred from
            the `dataclass`.

    Returns:
        A representation of the dataclass.
    """
    if default_values is None:
        default_values = {}
    non_default_values = {}

    def equals(value, default_value):
        if value is None and default_value is None:
            return True
        if value is None or default_value is None:
            return False
        return value == default_value
    for field in fields(obj):
        value = getattr(obj, field.name)
        default_value = default_values.get(field.name, field.default)
        is_required = isinstance(field.default, _MISSING_TYPE)
        if is_required or not equals(value, default_value):
            non_default_values[field.name] = value
    string = f'{obj.__class__.__name__}('
    string += ', '.join((f'{name}={value!r}' for name, value in non_default_values.items()))
    string += ')'
    return string