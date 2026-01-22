from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesCanaryevaluationsGetRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesCanaryevaluationsGetRequest object.

  Fields:
    name: Required. Name of the CanaryEvaluation. Use the following structure
      in your request:
      `organizations/{org}/instances/*/canaryevaluations/{evaluation}`
  """
    name = _messages.StringField(1, required=True)