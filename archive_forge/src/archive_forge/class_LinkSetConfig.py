from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkSetConfig(_messages.Message):
    """A LinkSetConfig object.

  Fields:
    assetType: Required. Reference to the asset type name for the type of the
      assets in this set, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{name}`.
  """
    assetType = _messages.StringField(1)