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
def _AddPrivateHelperMethods(message_descriptor, cls):
    """Adds implementation of private helper methods to cls."""

    def Modified(self):
        """Sets the _cached_byte_size_dirty bit to true,
    and propagates this to our listener iff this was a state change.
    """
        if not self._cached_byte_size_dirty:
            self._cached_byte_size_dirty = True
            self._listener_for_children.dirty = True
            self._is_present_in_parent = True
            self._listener.Modified()

    def _UpdateOneofState(self, field):
        """Sets field as the active field in its containing oneof.

    Will also delete currently active field in the oneof, if it is different
    from the argument. Does not mark the message as modified.
    """
        other_field = self._oneofs.setdefault(field.containing_oneof, field)
        if other_field is not field:
            del self._fields[other_field]
            self._oneofs[field.containing_oneof] = field
    cls._Modified = Modified
    cls.SetInParent = Modified
    cls._UpdateOneofState = _UpdateOneofState