from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaListSubscriptionsResponse(_messages.Message):
    """Response message for BeyondCorp.ListSubscriptions.

  Fields:
    nextPageToken: A token to retrieve the next page of results, or empty if
      there are no more results in the list.
    subscriptions: A list of BeyondCorp Subscriptions in the organization.
  """
    nextPageToken = _messages.StringField(1)
    subscriptions = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription', 2, repeated=True)