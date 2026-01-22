from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsTopicsSubscriptionsListRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsTopicsSubscriptionsListRequest object.

  Fields:
    name: Required. The name of the topic whose subscriptions to list.
    pageSize: The maximum number of subscriptions to return. The service may
      return fewer than this value. If unset or zero, all subscriptions for
      the given topic will be returned.
    pageToken: A page token, received from a previous `ListTopicSubscriptions`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListTopicSubscriptions` must match the
      call that provided the page token.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)