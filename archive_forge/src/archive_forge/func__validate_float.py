import array
import numbers
from collections.abc import Mapping, Sequence
from typing import Any, Iterable
from .const import INT_MAX_VALUE, INT_MIN_VALUE, LONG_MAX_VALUE, LONG_MIN_VALUE
from ._validate_common import ValidationError, ValidationErrorData
from .schema import extract_record_type, extract_logical_type, schema_name, parse_schema
from .logical_writers import LOGICAL_WRITERS
from ._schema_common import UnknownType
from .types import Schema, NamedSchemas
def _validate_float(datum, **kwargs):
    """
    Check that the data value is a floating
    point number or double precision.

    conditional python types
    (int, float, numbers.Real)
    """
    return isinstance(datum, (int, float, numbers.Real)) and (not isinstance(datum, bool))