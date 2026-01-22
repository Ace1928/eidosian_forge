from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisSetIamPolicyRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisSetIamPolicyRequest object.

  Fields:
    apigatewaySetIamPolicyRequest: A ApigatewaySetIamPolicyRequest resource to
      be passed as the request body.
    resource: REQUIRED: The resource for which the policy is being specified.
      See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    apigatewaySetIamPolicyRequest = _messages.MessageField('ApigatewaySetIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)