from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1VectorConfig(_messages.Message):
    """The index configuration to support vector search operations

  Fields:
    dimension: Required. The vector dimension this configuration applies to.
      The resulting index will only include vectors of this dimension, and can
      be used for vector search with the same dimension.
    flat: Indicates the vector index is a flat index.
  """
    dimension = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    flat = _messages.MessageField('GoogleFirestoreAdminV1FlatIndex', 2)