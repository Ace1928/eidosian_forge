from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import clusters
from googlecloudsdk.api_lib.vmware import networks
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.exceptions import Error
def UnDelete(self, resource):
    request = self.messages.VmwareengineProjectsLocationsPrivateCloudsUndeleteRequest(name=resource.RelativeName())
    return self.service.Undelete(request)