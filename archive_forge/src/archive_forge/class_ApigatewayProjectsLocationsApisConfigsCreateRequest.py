from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisConfigsCreateRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisConfigsCreateRequest object.

  Fields:
    apiConfigId: Required. Identifier to assign to the API Config. Must be
      unique within scope of the parent resource.
    apigatewayApiConfig: A ApigatewayApiConfig resource to be passed as the
      request body.
    parent: Required. Parent resource of the API Config, of the form:
      `projects/*/locations/global/apis/*`
  """
    apiConfigId = _messages.StringField(1)
    apigatewayApiConfig = _messages.MessageField('ApigatewayApiConfig', 2)
    parent = _messages.StringField(3, required=True)