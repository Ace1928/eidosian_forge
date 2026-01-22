from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def _InternalUnpackAny(msg):
    """Unpacks Any message and returns the unpacked message.

  This internal method is different from public Any Unpack method which takes
  the target message as argument. _InternalUnpackAny method does not have
  target message type and need to find the message type in descriptor pool.

  Args:
    msg: An Any message to be unpacked.

  Returns:
    The unpacked message.
  """
    from google.protobuf import symbol_database
    factory = symbol_database.Default()
    type_url = msg.type_url
    if not type_url:
        return None
    type_name = type_url.split('/')[-1]
    descriptor = factory.pool.FindMessageTypeByName(type_name)
    if descriptor is None:
        return None
    message_class = factory.GetPrototype(descriptor)
    message = message_class()
    message.ParseFromString(msg.value)
    return message