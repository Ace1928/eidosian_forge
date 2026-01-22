from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGatewaysCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGatewaysCreateRequest object.

  Fields:
    gateway: A Gateway resource to be passed as the request body.
    gatewayId: Required. Short name of the Gateway resource to be created.
    parent: Required. The parent resource of the Gateway. Must be in the
      format `projects/*/locations/*`.
  """
    gateway = _messages.MessageField('Gateway', 1)
    gatewayId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)