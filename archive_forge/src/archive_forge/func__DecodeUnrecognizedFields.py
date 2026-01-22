import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _DecodeUnrecognizedFields(message, pair_type):
    """Process unrecognized fields in message."""
    new_values = []
    codec = _ProtoJsonApiTools.Get()
    for unknown_field in message.all_unrecognized_fields():
        value, _ = message.get_unrecognized_field_info(unknown_field)
        value_type = pair_type.field_by_name('value')
        if isinstance(value_type, messages.MessageField):
            decoded_value = DictToMessage(value, pair_type.value.message_type)
        else:
            decoded_value = codec.decode_field(pair_type.value, value)
        try:
            new_pair_key = str(unknown_field)
        except UnicodeEncodeError:
            new_pair_key = protojson.ProtoJson().decode_field(pair_type.key, unknown_field)
        new_pair = pair_type(key=new_pair_key, value=decoded_value)
        new_values.append(new_pair)
    return new_values