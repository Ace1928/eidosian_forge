from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceContext(_messages.Message):
    """`SourceContext` represents information about the source of a protobuf
  element, like the file in which it is defined.

  Fields:
    fileName: The path-qualified name of the .proto file that contained the
      associated protobuf element.  For example:
      `"google/protobuf/source_context.proto"`.
  """
    fileName = _messages.StringField(1)