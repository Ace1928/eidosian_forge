import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class OneofDescriptor(DescriptorBase):
    """Descriptor for a oneof field.

  Attributes:
    name (str): Name of the oneof field.
    full_name (str): Full name of the oneof field, including package name.
    index (int): 0-based index giving the order of the oneof field inside
      its containing type.
    containing_type (Descriptor): :class:`Descriptor` of the protocol message
      type that contains this field.  Set by the :class:`Descriptor` constructor
      if we're passed into one.
    fields (list[FieldDescriptor]): The list of field descriptors this
      oneof can contain.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.OneofDescriptor

        def __new__(cls, name, full_name, index, containing_type, fields, options=None, serialized_options=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return _message.default_pool.FindOneofByName(full_name)

    def __init__(self, name, full_name, index, containing_type, fields, options=None, serialized_options=None, create_key=None):
        """Arguments are as described in the attribute description above."""
        if create_key is not _internal_create_key:
            _Deprecated('OneofDescriptor')
        super(OneofDescriptor, self).__init__(containing_type.file if containing_type else None, options, serialized_options, 'OneofOptions')
        self.name = name
        self.full_name = full_name
        self.index = index
        self.containing_type = containing_type
        self.fields = fields

    @property
    def _parent(self):
        return self.containing_type