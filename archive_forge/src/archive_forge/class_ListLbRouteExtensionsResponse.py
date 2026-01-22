from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLbRouteExtensionsResponse(_messages.Message):
    """Message for response to listing `LbRouteExtension` resources.

  Fields:
    lbRouteExtensions: The list of `LbRouteExtension` resources.
    nextPageToken: A token identifying a page of results that the server
      returns.
    unreachable: Locations that could not be reached.
  """
    lbRouteExtensions = _messages.MessageField('LbRouteExtension', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)