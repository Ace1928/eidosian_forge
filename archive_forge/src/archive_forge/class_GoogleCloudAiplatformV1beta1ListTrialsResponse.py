from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListTrialsResponse(_messages.Message):
    """Response message for VizierService.ListTrials.

  Fields:
    nextPageToken: Pass this token as the `page_token` field of the request
      for a subsequent call. If this field is omitted, there are no subsequent
      pages.
    trials: The Trials associated with the Study.
  """
    nextPageToken = _messages.StringField(1)
    trials = _messages.MessageField('GoogleCloudAiplatformV1beta1Trial', 2, repeated=True)