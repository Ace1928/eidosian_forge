from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeOrganizationSecurityPoliciesGetRequest(_messages.Message):
    """A ComputeOrganizationSecurityPoliciesGetRequest object.

  Fields:
    securityPolicy: Name of the security policy to get.
  """
    securityPolicy = _messages.StringField(1, required=True)