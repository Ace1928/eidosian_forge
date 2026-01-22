from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessageSubscriptionsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessageSubscriptionsDeleteRequest
  object.

  Fields:
    name: Required. A name of the MessageSubscription to delete. Must be in
      the format `projects/*/locations/*/messageSubscriptions/*`.
  """
    name = _messages.StringField(1, required=True)