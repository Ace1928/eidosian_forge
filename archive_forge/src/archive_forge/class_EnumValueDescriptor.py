import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class EnumValueDescriptor(DescriptorBase):
    """Descriptor for a single value within an enum.

  Attributes:
    name (str): Name of this value.
    index (int): Dense, 0-indexed index giving the order that this
      value appears textually within its enum in the .proto file.
    number (int): Actual number assigned to this enum value.
    type (EnumDescriptor): :class:`EnumDescriptor` to which this value
      belongs.  Set by :class:`EnumDescriptor`'s constructor if we're
      passed into one.
    options (descriptor_pb2.EnumValueOptions): Enum value options message or
      None to use default enum value options options.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.EnumValueDescriptor

        def __new__(cls, name, index, number, type=None, options=None, serialized_options=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return None

    def __init__(self, name, index, number, type=None, options=None, serialized_options=None, create_key=None):
        """Arguments are as described in the attribute description above."""
        if create_key is not _internal_create_key:
            _Deprecated('EnumValueDescriptor')
        super(EnumValueDescriptor, self).__init__(type.file if type else None, options, serialized_options, 'EnumValueOptions')
        self.name = name
        self.index = index
        self.number = number
        self.type = type

    @property
    def _parent(self):
        return self.type