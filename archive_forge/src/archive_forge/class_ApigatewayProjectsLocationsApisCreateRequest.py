from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisCreateRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisCreateRequest object.

  Fields:
    apiId: Required. Identifier to assign to the API. Must be unique within
      scope of the parent resource.
    apigatewayApi: A ApigatewayApi resource to be passed as the request body.
    parent: Required. Parent resource of the API, of the form:
      `projects/*/locations/global`
  """
    apiId = _messages.StringField(1)
    apigatewayApi = _messages.MessageField('ApigatewayApi', 2)
    parent = _messages.StringField(3, required=True)