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
class _OneofListener(_Listener):
    """Special listener implementation for setting composite oneof fields."""

    def __init__(self, parent_message, field):
        """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
      field: The descriptor of the field being set in the parent message.
    """
        super(_OneofListener, self).__init__(parent_message)
        self._field = field

    def Modified(self):
        """Also updates the state of the containing oneof in the parent message."""
        try:
            self._parent_message_weakref._UpdateOneofState(self._field)
            super(_OneofListener, self).Modified()
        except ReferenceError:
            pass