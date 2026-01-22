from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DatasetSpec(_messages.Message):
    """Specification that applies to a dataset. Valid only for entries with the
  `DATASET` type.

  Fields:
    vertexDatasetSpec: Vertex AI Dataset specific fields
  """
    vertexDatasetSpec = _messages.MessageField('GoogleCloudDatacatalogV1VertexDatasetSpec', 1)