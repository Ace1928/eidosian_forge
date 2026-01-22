from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtoDescriptor(_messages.Message):
    """Contains a serialized protoc-generated protocol buffer message
  descriptor set along with a URL that describes the type of the descriptor
  message.

  Fields:
    typeUrl: A URL/resource name whose content describes the type of the
      serialized protocol buffer message.  Only
      'type.googleapis.com/google.protobuf.FileDescriptorSet' is supported. If
      the type_url is not specificed,
      'type.googleapis.com/google.protobuf.FileDescriptorSet' will be assumed.
    value: Must be a valid serialized protocol buffer descriptor set.  To
      generate, use protoc with imports and source info included. For an
      example test.proto file, the following command would put the value in a
      new file named descriptor.pb.  $protoc --include_imports
      --include_source_info test.proto -o descriptor.pb
  """
    typeUrl = _messages.StringField(1)
    value = _messages.BytesField(2)