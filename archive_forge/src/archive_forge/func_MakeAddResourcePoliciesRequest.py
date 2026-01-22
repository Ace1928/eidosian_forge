from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def MakeAddResourcePoliciesRequest(self, resource_policies, client_to_make_request):
    add_request = self._messages.ComputeRegionDisksAddResourcePoliciesRequest(disk=self._disk_ref.Name(), project=self._disk_ref.project, region=self._disk_ref.region, regionDisksAddResourcePoliciesRequest=self._messages.RegionDisksAddResourcePoliciesRequest(resourcePolicies=resource_policies))
    return client_to_make_request.MakeRequests([(self._client.regionDisks, 'AddResourcePolicies', add_request)])