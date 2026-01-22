from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageConsumerPoliciesGetRequest(_messages.Message):
    """A ServiceusageConsumerPoliciesGetRequest object.

  Fields:
    name: Required. The name of the consumer policy to retrieve. Format:
      `projects/100/consumerPolicies/default`,
      `folders/101/consumerPolicies/default`,
      `organizations/102/consumerPolicies/default`.
  """
    name = _messages.StringField(1, required=True)