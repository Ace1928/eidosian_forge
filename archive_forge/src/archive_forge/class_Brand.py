from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Brand(_messages.Message):
    """OAuth brand data. NOTE: Only contains a portion of the data that
  describes a brand.

  Fields:
    applicationTitle: Application name displayed on OAuth consent screen.
    name: Output only. Identifier of the brand. NOTE: GCP project number
      achieves the same brand identification purpose as only one brand per
      project can be created.
    orgInternalOnly: Output only. Whether the brand is only intended for usage
      inside the G Suite organization only.
    supportEmail: Support email displayed on the OAuth consent screen.
  """
    applicationTitle = _messages.StringField(1)
    name = _messages.StringField(2)
    orgInternalOnly = _messages.BooleanField(3)
    supportEmail = _messages.StringField(4)