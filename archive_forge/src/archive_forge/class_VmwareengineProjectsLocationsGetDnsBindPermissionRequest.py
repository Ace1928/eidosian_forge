from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsGetDnsBindPermissionRequest(_messages.Message):
    """A VmwareengineProjectsLocationsGetDnsBindPermissionRequest object.

  Fields:
    name: Required. The name of the resource which stores the users/service
      accounts having the permission to bind to the corresponding intranet VPC
      of the consumer project. DnsBindPermission is a global resource.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/global/dnsBindPermission`
  """
    name = _messages.StringField(1, required=True)