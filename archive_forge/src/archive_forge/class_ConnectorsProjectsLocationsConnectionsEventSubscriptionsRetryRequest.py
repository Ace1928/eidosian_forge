from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsEventSubscriptionsRetryRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsEventSubscriptionsRetryRequest
  object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/connections/*/eventSubscriptions/*`
    retryEventSubscriptionRequest: A RetryEventSubscriptionRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    retryEventSubscriptionRequest = _messages.MessageField('RetryEventSubscriptionRequest', 2)