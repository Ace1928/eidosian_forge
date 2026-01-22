from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelatedResource(_messages.Message):
    """The detailed related resource.

  Fields:
    assetType: The type of the asset. Example:
      `compute.googleapis.com/Instance`
    fullResourceName: The full resource name of the related resource. Example:
      `//compute.googleapis.com/projects/my_proj_123/zones/instance/instance12
      3`
  """
    assetType = _messages.StringField(1)
    fullResourceName = _messages.StringField(2)