from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsGatewaySecurityPoliciesGetRequest
  object.

  Fields:
    name: Required. A name of the GatewaySecurityPolicy to get. Must be in the
      format
      `projects/{project}/locations/{location}/gatewaySecurityPolicies/*`.
  """
    name = _messages.StringField(1, required=True)