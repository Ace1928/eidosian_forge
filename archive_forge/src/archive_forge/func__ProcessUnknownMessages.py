import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _ProcessUnknownMessages(message, encoded_message):
    """Store any remaining unknown fields as strings.

    ProtoRPC currently ignores unknown values for which no type can be
    determined (and logs a "No variant found" message). For the purposes
    of reserializing, this is quite harmful (since it throws away
    information). Here we simply add those as unknown fields of type
    string (so that they can easily be reserialized).

    Args:
      message: Proto message we've decoded thus far.
      encoded_message: JSON string we're decoding.

    Returns:
      message, with any remaining unrecognized fields saved.
    """
    if not encoded_message:
        return message
    decoded_message = json.loads(six.ensure_str(encoded_message))
    message_fields = [x.name for x in message.all_fields()] + list(message.all_unrecognized_fields())
    missing_fields = [x for x in decoded_message.keys() if x not in message_fields]
    for field_name in missing_fields:
        message.set_unrecognized_field(field_name, decoded_message[field_name], messages.Variant.STRING)
    return message