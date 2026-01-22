import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _MapItems(message, field):
    """Yields the (key, value) pair of the map values."""
    assert _IsMap(message, field)
    map_message = message.get_assigned_value(field.name)
    additional_properties = map_message.get_assigned_value('additionalProperties')
    for kv_pair in additional_properties:
        yield (kv_pair.key, kv_pair.value)