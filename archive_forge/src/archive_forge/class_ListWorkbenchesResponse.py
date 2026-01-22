from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkbenchesResponse(_messages.Message):
    """Message for response to listing Workbenches

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
    workbenches: The list of Workbench
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    workbenches = _messages.MessageField('Workbench', 3, repeated=True)