from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesDeleteRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the external access firewall rule to
      delete. Resource names are schemeless URIs that follow the conventions
      in https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-
      policy/externalAccessRules/my-rule`
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check if the
      original operation with the same request ID was received, and if so,
      will ignore the second request. This prevents clients from accidentally
      creating duplicate commitments. The request ID must be a valid UUID with
      the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)