from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListZonesResponse(_messages.Message):
    """Deprecated: not implemented. Message for response to listing Zones

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
    zones: The list of Zone
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    zones = _messages.MessageField('Zone', 3, repeated=True)