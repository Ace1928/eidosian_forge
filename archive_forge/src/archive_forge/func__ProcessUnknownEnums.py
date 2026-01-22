import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _ProcessUnknownEnums(message, encoded_message):
    """Add unknown enum values from encoded_message as unknown fields.

    ProtoRPC diverges from the usual protocol buffer behavior here and
    doesn't allow unknown fields. Throwing on unknown fields makes it
    impossible to let servers add new enum values and stay compatible
    with older clients, which isn't reasonable for us. We simply store
    unrecognized enum values as unknown fields, and all is well.

    Args:
      message: Proto message we've decoded thus far.
      encoded_message: JSON string we're decoding.

    Returns:
      message, with any unknown enums stored as unrecognized fields.
    """
    if not encoded_message:
        return message
    decoded_message = json.loads(six.ensure_str(encoded_message))
    for field in message.all_fields():
        if isinstance(field, messages.EnumField) and field.name in decoded_message and (message.get_assigned_value(field.name) is None):
            message.set_unrecognized_field(field.name, decoded_message[field.name], messages.Variant.ENUM)
    return message