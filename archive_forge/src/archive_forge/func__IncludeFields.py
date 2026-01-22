import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _IncludeFields(encoded_message, message, include_fields):
    """Add the requested fields to the encoded message."""
    if include_fields is None:
        return encoded_message
    result = json.loads(encoded_message)
    for field_name in include_fields:
        try:
            value = _GetField(message, field_name.split('.'))
            nullvalue = None
            if isinstance(value, list):
                nullvalue = []
        except KeyError:
            raise exceptions.InvalidDataError('No field named %s in message of type %s' % (field_name, type(message)))
        _SetField(result, field_name.split('.'), nullvalue)
    return json.dumps(result)