from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UpsertDatapointsRequest(_messages.Message):
    """Request message for IndexService.UpsertDatapoints

  Fields:
    datapoints: A list of datapoints to be created/updated.
    updateMask: Optional. Update mask is used to specify the fields to be
      overwritten in the datapoints by the update. The fields specified in the
      update_mask are relative to each IndexDatapoint inside datapoints, not
      the full request. Updatable fields: * Use `all_restricts` to update both
      restricts and numeric_restricts.
  """
    datapoints = _messages.MessageField('GoogleCloudAiplatformV1IndexDatapoint', 1, repeated=True)
    updateMask = _messages.StringField(2)