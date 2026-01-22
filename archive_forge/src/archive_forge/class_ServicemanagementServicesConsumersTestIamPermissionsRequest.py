from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConsumersTestIamPermissionsRequest(_messages.Message):
    """A ServicemanagementServicesConsumersTestIamPermissionsRequest object.

  Fields:
    consumersId: Part of `resource`. See documentation of `servicesId`.
    servicesId: Part of `resource`. REQUIRED: The resource for which the
      policy detail is being requested. See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
    testIamPermissionsRequest: A TestIamPermissionsRequest resource to be
      passed as the request body.
  """
    consumersId = _messages.StringField(1, required=True)
    servicesId = _messages.StringField(2, required=True)
    testIamPermissionsRequest = _messages.MessageField('TestIamPermissionsRequest', 3)