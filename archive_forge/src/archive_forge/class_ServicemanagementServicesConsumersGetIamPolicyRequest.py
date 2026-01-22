from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConsumersGetIamPolicyRequest(_messages.Message):
    """A ServicemanagementServicesConsumersGetIamPolicyRequest object.

  Fields:
    consumersId: Part of `resource`. See documentation of `servicesId`.
    getIamPolicyRequest: A GetIamPolicyRequest resource to be passed as the
      request body.
    servicesId: Part of `resource`. REQUIRED: The resource for which the
      policy is being requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    consumersId = _messages.StringField(1, required=True)
    getIamPolicyRequest = _messages.MessageField('GetIamPolicyRequest', 2)
    servicesId = _messages.StringField(3, required=True)