from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsSubscriptionsListRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsSubscriptionsListRequest object.

  Fields:
    pageSize: The maximum number of subscriptions to return. The service may
      return fewer than this value. If unset or zero, all subscriptions for
      the parent will be returned.
    pageToken: A page token, received from a previous `ListSubscriptions`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListSubscriptions` must match the call
      that provided the page token.
    parent: Required. The parent whose subscriptions are to be listed.
      Structured like `projects/{project_number}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)