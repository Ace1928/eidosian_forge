from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Owner(_messages.Message):
    """The owner of the asset, set by the system.

  Fields:
    linkId: The link ID in the owner asset.
    ownerAsset: The name of the owner asset.
  """
    linkId = _messages.StringField(1)
    ownerAsset = _messages.StringField(2)