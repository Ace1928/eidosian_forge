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
def _AddWhichOneofMethod(message_descriptor, cls):

    def WhichOneof(self, oneof_name):
        """Returns the name of the currently set field inside a oneof, or None."""
        try:
            field = message_descriptor.oneofs_by_name[oneof_name]
        except KeyError:
            raise ValueError('Protocol message has no oneof "%s" field.' % oneof_name)
        nested_field = self._oneofs.get(field, None)
        if nested_field is not None and self.HasField(nested_field.name):
            return nested_field.name
        else:
            return None
    cls.WhichOneof = WhichOneof