from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsOperationsCancelRequest(_messages.Message):
    """A ApigatewayProjectsLocationsOperationsCancelRequest object.

  Fields:
    apigatewayCancelOperationRequest: A ApigatewayCancelOperationRequest
      resource to be passed as the request body.
    name: The name of the operation resource to be cancelled.
  """
    apigatewayCancelOperationRequest = _messages.MessageField('ApigatewayCancelOperationRequest', 1)
    name = _messages.StringField(2, required=True)