from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayListApiConfigsResponse(_messages.Message):
    """Response message for ApiGatewayService.ListApiConfigs

  Fields:
    apiConfigs: API Configs.
    nextPageToken: Next page token.
    unreachableLocations: Locations that could not be reached.
  """
    apiConfigs = _messages.MessageField('ApigatewayApiConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachableLocations = _messages.StringField(3, repeated=True)