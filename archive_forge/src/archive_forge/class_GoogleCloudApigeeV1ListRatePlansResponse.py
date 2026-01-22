from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListRatePlansResponse(_messages.Message):
    """Response for ListRatePlans.

  Fields:
    nextStartKey: Value that can be sent as `startKey` to retrieve the next
      page of content. If this field is omitted, there are no subsequent
      pages.
    ratePlans: List of rate plans in an organization.
  """
    nextStartKey = _messages.StringField(1)
    ratePlans = _messages.MessageField('GoogleCloudApigeeV1RatePlan', 2, repeated=True)