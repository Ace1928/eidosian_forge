from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResourceAllowancesResponse(_messages.Message):
    """ListResourceAllowances Response.

  Fields:
    nextPageToken: Next page token.
    resourceAllowances: ResourceAllowances.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    resourceAllowances = _messages.MessageField('ResourceAllowance', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)