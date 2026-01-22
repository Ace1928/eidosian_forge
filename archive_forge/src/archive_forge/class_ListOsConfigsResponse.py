from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListOsConfigsResponse(_messages.Message):
    """A response message for listing OsConfigs.

  Fields:
    nextPageToken: A pagination token that can be used to get the next page of
      OsConfigs.
    osConfigs: The list of OsConfigs.
  """
    nextPageToken = _messages.StringField(1)
    osConfigs = _messages.MessageField('OsConfig', 2, repeated=True)