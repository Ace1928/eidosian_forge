from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexConfig(_messages.Message):
    """Configuration for an indexed field.

  Enums:
    TypeValueValuesEnum: Required. The type of data in this index.

  Fields:
    createTime: Output only. The timestamp when the index was last
      modified.This is used to return the timestamp, and will be ignored if
      supplied during update.
    fieldPath: Required. The LogEntry field path to index.Note that some paths
      are automatically indexed, and other paths are not eligible for
      indexing. See indexing documentation(
      https://cloud.google.com/logging/docs/view/advanced-queries#indexed-
      fields) for details.For example: jsonPayload.request.status
    type: Required. The type of data in this index.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of data in this index.

    Values:
      INDEX_TYPE_UNSPECIFIED: The index's type is unspecified.
      INDEX_TYPE_STRING: The index is a string-type index.
      INDEX_TYPE_INTEGER: The index is a integer-type index.
    """
        INDEX_TYPE_UNSPECIFIED = 0
        INDEX_TYPE_STRING = 1
        INDEX_TYPE_INTEGER = 2
    createTime = _messages.StringField(1)
    fieldPath = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)