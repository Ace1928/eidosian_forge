from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthosObservabilityFeatureSpec(_messages.Message):
    """**Anthos Observability**: Spec

  Fields:
    defaultMembershipSpec: Default membership spec for unconfigured
      memberships
  """
    defaultMembershipSpec = _messages.MessageField('AnthosObservabilityMembershipSpec', 1)