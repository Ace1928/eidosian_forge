from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
class _RegionalDisk(object):
    """A wrapper for Compute Engine RegionDisksService API client."""

    def __init__(self, client, disk_ref, messages):
        self._disk_ref = disk_ref
        self._client = client
        self._service = client.regionDisks
        self._messages = messages

    @classmethod
    def GetOperationCollection(cls):
        return 'compute.regionOperations'

    def GetService(self):
        return self._service

    def GetDiskRequestMessage(self):
        return self._messages.ComputeRegionDisksGetRequest(**self._disk_ref.AsDict())

    def GetDiskResource(self):
        request_msg = self.GetDiskRequestMessage()
        return self._service.Get(request_msg)

    def GetSetLabelsRequestMessage(self):
        return self._messages.RegionSetLabelsRequest

    def GetSetDiskLabelsRequestMessage(self, disk, labels):
        req = self._messages.ComputeRegionDisksSetLabelsRequest
        return req(project=self._disk_ref.project, resource=self._disk_ref.disk, region=self._disk_ref.region, regionSetLabelsRequest=self._messages.RegionSetLabelsRequest(labelFingerprint=disk.labelFingerprint, labels=labels))

    def GetDiskRegionName(self):
        return self._disk_ref.region

    def MakeAddResourcePoliciesRequest(self, resource_policies, client_to_make_request):
        add_request = self._messages.ComputeRegionDisksAddResourcePoliciesRequest(disk=self._disk_ref.Name(), project=self._disk_ref.project, region=self._disk_ref.region, regionDisksAddResourcePoliciesRequest=self._messages.RegionDisksAddResourcePoliciesRequest(resourcePolicies=resource_policies))
        return client_to_make_request.MakeRequests([(self._client.regionDisks, 'AddResourcePolicies', add_request)])

    def MakeRemoveResourcePoliciesRequest(self, resource_policies, client_to_make_request):
        remove_request = self._messages.ComputeRegionDisksRemoveResourcePoliciesRequest(disk=self._disk_ref.Name(), project=self._disk_ref.project, region=self._disk_ref.region, regionDisksRemoveResourcePoliciesRequest=self._messages.RegionDisksRemoveResourcePoliciesRequest(resourcePolicies=resource_policies))
        return client_to_make_request.MakeRequests([(self._client.regionDisks, 'RemoveResourcePolicies', remove_request)])