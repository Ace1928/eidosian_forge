from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPoliciesListPreconfiguredExpressionSetsResponse(_messages.Message):
    """A SecurityPoliciesListPreconfiguredExpressionSetsResponse object.

  Fields:
    preconfiguredExpressionSets: A SecurityPoliciesWafConfig attribute.
  """
    preconfiguredExpressionSets = _messages.MessageField('SecurityPoliciesWafConfig', 1)