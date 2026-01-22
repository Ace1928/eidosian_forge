from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSecuritygatewaysV1alphaListSecurityGatewaysResponse(_messages.Message):
    """Message for response to listing SecurityGateways.

  Fields:
    nextPageToken: A token to retrieve the next page of results, or empty if
      there are no more results in the list.
    securityGateways: A list of BeyondCorp SecurityGateway in the project.
    unreachable: A list of locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    securityGateways = _messages.MessageField('GoogleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)