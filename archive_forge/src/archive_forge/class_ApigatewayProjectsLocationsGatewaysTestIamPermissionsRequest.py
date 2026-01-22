from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsGatewaysTestIamPermissionsRequest(_messages.Message):
    """A ApigatewayProjectsLocationsGatewaysTestIamPermissionsRequest object.

  Fields:
    apigatewayTestIamPermissionsRequest: A ApigatewayTestIamPermissionsRequest
      resource to be passed as the request body.
    resource: REQUIRED: The resource for which the policy detail is being
      requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    apigatewayTestIamPermissionsRequest = _messages.MessageField('ApigatewayTestIamPermissionsRequest', 1)
    resource = _messages.StringField(2, required=True)