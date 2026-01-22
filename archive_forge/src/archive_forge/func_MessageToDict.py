import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def MessageToDict(message, always_print_fields_with_no_presence=False, preserving_proto_field_name=False, use_integers_for_enums=False, descriptor_pool=None, float_precision=None):
    """Converts protobuf message to a dictionary.

  When the dictionary is encoded to JSON, it conforms to proto3 JSON spec.

  Args:
    message: The protocol buffers message instance to serialize.
    always_print_fields_with_no_presence: If True, fields without
      presence (implicit presence scalars, repeated fields, and map fields) will
      always be serialized. Any field that supports presence is not affected by
      this option (including singular message fields and oneof fields).
    preserving_proto_field_name: If True, use the original proto field names as
      defined in the .proto file. If False, convert the field names to
      lowerCamelCase.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    float_precision: If set, use this to specify float field valid digits.

  Returns:
    A dict representation of the protocol buffer message.
  """
    printer = _Printer(preserving_proto_field_name, use_integers_for_enums, descriptor_pool, float_precision, always_print_fields_with_no_presence)
    return printer._MessageToJsonObject(message)