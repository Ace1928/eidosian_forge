from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConsumersSetIamPolicyRequest(_messages.Message):
    """A ServicemanagementServicesConsumersSetIamPolicyRequest object.

  Fields:
    consumersId: Part of `resource`. See documentation of `servicesId`.
    servicesId: Part of `resource`. REQUIRED: The resource for which the
      policy is being specified. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
    setIamPolicyRequest: A SetIamPolicyRequest resource to be passed as the
      request body.
  """
    consumersId = _messages.StringField(1, required=True)
    servicesId = _messages.StringField(2, required=True)
    setIamPolicyRequest = _messages.MessageField('SetIamPolicyRequest', 3)