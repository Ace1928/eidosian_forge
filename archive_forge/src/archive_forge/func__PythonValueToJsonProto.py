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
def _PythonValueToJsonProto(py_value):
    if isinstance(py_value, dict):
        return _PythonValueToJsonObject(py_value)
    if isinstance(py_value, Iterable) and (not isinstance(py_value, six.string_types)):
        return _PythonValueToJsonArray(py_value)
    return _PythonValueToJsonValue(py_value)