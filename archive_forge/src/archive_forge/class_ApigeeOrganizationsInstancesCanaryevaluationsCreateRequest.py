from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesCanaryevaluationsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesCanaryevaluationsCreateRequest object.

  Fields:
    googleCloudApigeeV1CanaryEvaluation: A GoogleCloudApigeeV1CanaryEvaluation
      resource to be passed as the request body.
    parent: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}/instances/{instance}`.
  """
    googleCloudApigeeV1CanaryEvaluation = _messages.MessageField('GoogleCloudApigeeV1CanaryEvaluation', 1)
    parent = _messages.StringField(2, required=True)