from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPeeringRoutesResponse(_messages.Message):
    """Response message for VmwareEngine.ListPeeringRoutes

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    peeringRoutes: A list of peering routes.
  """
    nextPageToken = _messages.StringField(1)
    peeringRoutes = _messages.MessageField('PeeringRoute', 2, repeated=True)