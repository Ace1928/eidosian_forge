from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ModelSpec(_messages.Message):
    """Specification that applies to a model. Valid only for entries with the
  `MODEL` type.

  Fields:
    vertexModelSpec: Specification for vertex model resources.
  """
    vertexModelSpec = _messages.MessageField('GoogleCloudDatacatalogV1VertexModelSpec', 1)