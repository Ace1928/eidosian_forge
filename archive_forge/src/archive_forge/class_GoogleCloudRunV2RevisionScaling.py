from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2RevisionScaling(_messages.Message):
    """Settings for revision-level scaling settings.

  Fields:
    maxInstanceCount: Optional. Maximum number of serving instances that this
      resource should have.
    minInstanceCount: Optional. Minimum number of serving instances that this
      resource should have.
  """
    maxInstanceCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minInstanceCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)