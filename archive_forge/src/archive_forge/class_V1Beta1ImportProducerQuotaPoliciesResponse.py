from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ImportProducerQuotaPoliciesResponse(_messages.Message):
    """Response message for ImportProducerQuotaPolicies

  Fields:
    policies: The policies that were created from the imported data.
  """
    policies = _messages.MessageField('V1Beta1ProducerQuotaPolicy', 1, repeated=True)