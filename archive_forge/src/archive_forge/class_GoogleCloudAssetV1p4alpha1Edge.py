from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudAssetV1p4alpha1Edge(_messages.Message):
    """A directional edge.

  Fields:
    sourceNode: The source node of the edge.
    targetNode: The target node of the edge.
  """
    sourceNode = _messages.StringField(1)
    targetNode = _messages.StringField(2)