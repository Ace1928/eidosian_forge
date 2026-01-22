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
def _AddMessageMethods(message_descriptor, cls):
    """Adds implementations of all Message methods to cls."""
    _AddListFieldsMethod(message_descriptor, cls)
    _AddHasFieldMethod(message_descriptor, cls)
    _AddClearFieldMethod(message_descriptor, cls)
    if message_descriptor.is_extendable:
        _AddClearExtensionMethod(cls)
        _AddHasExtensionMethod(cls)
    _AddEqualsMethod(message_descriptor, cls)
    _AddStrMethod(message_descriptor, cls)
    _AddReprMethod(message_descriptor, cls)
    _AddUnicodeMethod(message_descriptor, cls)
    _AddByteSizeMethod(message_descriptor, cls)
    _AddSerializeToStringMethod(message_descriptor, cls)
    _AddSerializePartialToStringMethod(message_descriptor, cls)
    _AddMergeFromStringMethod(message_descriptor, cls)
    _AddIsInitializedMethod(message_descriptor, cls)
    _AddMergeFromMethod(cls)
    _AddWhichOneofMethod(message_descriptor, cls)
    cls.Clear = _Clear
    cls.UnknownFields = _UnknownFields
    cls.DiscardUnknownFields = _DiscardUnknownFields
    cls._SetListener = _SetListener