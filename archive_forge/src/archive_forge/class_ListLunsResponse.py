from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLunsResponse(_messages.Message):
    """Response message containing the list of storage volume luns.

  Fields:
    luns: The list of luns.
    nextPageToken: A token identifying a page of results from the server.
    unreachable: Locations that could not be reached.
  """
    luns = _messages.MessageField('Lun', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)