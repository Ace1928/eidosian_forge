from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2ResourcePath(_messages.Message):
    """Represents the path of resources leading up to the resource this finding
  is about.

  Fields:
    nodes: The list of nodes that make the up resource path, ordered from
      lowest level to highest level.
  """
    nodes = _messages.MessageField('GoogleCloudSecuritycenterV2ResourcePathNode', 1, repeated=True)