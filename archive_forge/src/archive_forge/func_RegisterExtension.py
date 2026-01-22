from io import BytesIO
import struct
import sys
import weakref
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import enum_type_wrapper
from cloudsdk.google.protobuf.internal import extension_dict
from cloudsdk.google.protobuf.internal import message_listener as message_listener_mod
from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf.internal import well_known_types
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor as descriptor_mod
from cloudsdk.google.protobuf import message as message_mod
from cloudsdk.google.protobuf import text_format
def RegisterExtension(extension_handle):
    extension_handle.containing_type = cls.DESCRIPTOR
    cls.DESCRIPTOR.file.pool._AddExtensionDescriptor(extension_handle)
    _AttachFieldHelpers(cls, extension_handle)