from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesUnbindDeviceFromGatewayRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesUnbindDeviceFromGatewayRequest
  object.

  Fields:
    parent: Required. The name of the registry. For example,
      `projects/example-project/locations/us-central1/registries/my-registry`.
    unbindDeviceFromGatewayRequest: A UnbindDeviceFromGatewayRequest resource
      to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    unbindDeviceFromGatewayRequest = _messages.MessageField('UnbindDeviceFromGatewayRequest', 2)