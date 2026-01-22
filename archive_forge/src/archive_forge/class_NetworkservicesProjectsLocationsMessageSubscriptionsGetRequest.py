from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessageSubscriptionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessageSubscriptionsGetRequest object.

  Fields:
    name: Required. A name of the MessageSubscription to get. Must be in the
      format `projects/*/locations/*/messageSubscriptions/*`.
  """
    name = _messages.StringField(1, required=True)