from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsDnsBindPermissionGrantRequest(_messages.Message):
    """A VmwareengineProjectsLocationsDnsBindPermissionGrantRequest object.

  Fields:
    grantDnsBindPermissionRequest: A GrantDnsBindPermissionRequest resource to
      be passed as the request body.
    name: Required. The name of the resource which stores the users/service
      accounts having the permission to bind to the corresponding intranet VPC
      of the consumer project. DnsBindPermission is a global resource.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/global/dnsBindPermission`
  """
    grantDnsBindPermissionRequest = _messages.MessageField('GrantDnsBindPermissionRequest', 1)
    name = _messages.StringField(2, required=True)