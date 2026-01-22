from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesGetRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesGetRequest
  object.

  Fields:
    name: Required. The resource name of the external access firewall rule to
      retrieve. Resource names are schemeless URIs that follow the conventions
      in https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-
      policy/externalAccessRules/my-rule`
  """
    name = _messages.StringField(1, required=True)