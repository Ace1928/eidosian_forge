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
def _AddClearExtensionMethod(cls):
    """Helper for _AddMessageMethods()."""

    def ClearExtension(self, field_descriptor):
        extension_dict._VerifyExtensionHandle(self, field_descriptor)
        if field_descriptor in self._fields:
            del self._fields[field_descriptor]
        self._Modified()
    cls.ClearExtension = ClearExtension