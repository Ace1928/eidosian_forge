from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListFreeTrialsResponse(_messages.Message):
    """Response message for ConsumerProcurementService.ListFreeTrials.

  Fields:
    freeTrials: The list of FreeTrialss in this response.
    nextPageToken: The token for fetching the next page.
  """
    freeTrials = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial', 1, repeated=True)
    nextPageToken = _messages.StringField(2)