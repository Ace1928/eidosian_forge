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
def _JsonValueToPythonValue(json_value):
    """Convert the given JsonValue to a json string."""
    util.Typecheck(json_value, JsonValue)
    _ValidateJsonValue(json_value)
    if json_value.is_null:
        return None
    entries = [(f, json_value.get_assigned_value(f.name)) for f in json_value.all_fields()]
    assigned_entries = [(f, value) for f, value in entries if value is not None]
    field, value = assigned_entries[0]
    if not isinstance(field, messages.MessageField):
        return value
    elif field.message_type is JsonObject:
        return _JsonObjectToPythonValue(value)
    elif field.message_type is JsonArray:
        return _JsonArrayToPythonValue(value)