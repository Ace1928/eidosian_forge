from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworkPlacementsGetRequest(_messages.Message):
    """A ComputeNetworkPlacementsGetRequest object.

  Fields:
    networkPlacement: Name of the network placement to return.
    project: Project ID for this request.
  """
    networkPlacement = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)