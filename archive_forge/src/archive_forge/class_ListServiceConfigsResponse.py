from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceConfigsResponse(_messages.Message):
    """Response message for ListServiceConfigs method.

  Fields:
    nextPageToken: The token of the next page of results.
    serviceConfigs: The list of service config resources.
  """
    nextPageToken = _messages.StringField(1)
    serviceConfigs = _messages.MessageField('Service', 2, repeated=True)