from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeOrganizationSecurityPoliciesGetAssociationRequest(_messages.Message):
    """A ComputeOrganizationSecurityPoliciesGetAssociationRequest object.

  Fields:
    name: The name of the association to get from the security policy.
    securityPolicy: Name of the security policy to which the queried rule
      belongs.
  """
    name = _messages.StringField(1)
    securityPolicy = _messages.StringField(2, required=True)