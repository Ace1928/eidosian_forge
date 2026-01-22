import datetime
import json
import numbers
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def _PythonValueToJsonValue(py_value):
    """Convert the given python value to a JsonValue."""
    if py_value is None:
        return JsonValue(is_null=True)
    if isinstance(py_value, bool):
        return JsonValue(boolean_value=py_value)
    if isinstance(py_value, six.string_types):
        return JsonValue(string_value=py_value)
    if isinstance(py_value, numbers.Number):
        if isinstance(py_value, six.integer_types):
            if _MININT64 < py_value < _MAXINT64:
                return JsonValue(integer_value=py_value)
        return JsonValue(double_value=float(py_value))
    if isinstance(py_value, dict):
        return JsonValue(object_value=_PythonValueToJsonObject(py_value))
    if isinstance(py_value, Iterable):
        return JsonValue(array_value=_PythonValueToJsonArray(py_value))
    raise exceptions.InvalidDataError('Cannot convert "%s" to JsonValue' % py_value)