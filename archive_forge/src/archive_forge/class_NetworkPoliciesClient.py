from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.api_lib.vmware.networks import NetworksClient
class NetworkPoliciesClient(util.VmwareClientBase):
    """VMware Engine network policy client."""

    def __init__(self):
        super(NetworkPoliciesClient, self).__init__()
        self.service = self.client.projects_locations_networkPolicies
        self.networks_client = NetworksClient()

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsNetworkPoliciesGetRequest(name=resource.RelativeName())
        response = self.service.Get(request)
        return response

    def Create(self, resource, vmware_engine_network_id, edge_services_cidr, description=None, internet_access=None, external_ip_access=None):
        parent = resource.Parent().RelativeName()
        project = resource.Parent().Parent().Name()
        network_policy_id = resource.Name()
        network_policy = self.messages.NetworkPolicy(description=description)
        internet_access_obj = self.messages.NetworkService(enabled=internet_access)
        external_ip_access_obj = self.messages.NetworkService(enabled=external_ip_access)
        ven = self.networks_client.GetByID(project, vmware_engine_network_id)
        network_policy.vmwareEngineNetwork = ven.name
        network_policy.edgeServicesCidr = edge_services_cidr
        network_policy.internetAccess = internet_access_obj
        network_policy.externalIp = external_ip_access_obj
        request = self.messages.VmwareengineProjectsLocationsNetworkPoliciesCreateRequest(parent=parent, networkPolicy=network_policy, networkPolicyId=network_policy_id)
        return self.service.Create(request)

    def Update(self, resource, description=None, edge_services_cidr=None, internet_access=None, external_ip_access=None):
        network_policy = self.Get(resource)
        update_mask = []
        if description is not None:
            network_policy.description = description
            update_mask.append('description')
        if edge_services_cidr is not None:
            network_policy.edgeServicesCidr = edge_services_cidr
            update_mask.append('edge_services_cidr')
        if internet_access is not None:
            internet_access_obj = self.messages.NetworkService(enabled=internet_access)
            network_policy.internetAccess = internet_access_obj
            update_mask.append('internet_access.enabled')
        if external_ip_access is not None:
            external_ip_access_obj = self.messages.NetworkService(enabled=external_ip_access)
            network_policy.externalIp = external_ip_access_obj
            update_mask.append('external_ip.enabled')
        request = self.messages.VmwareengineProjectsLocationsNetworkPoliciesPatchRequest(networkPolicy=network_policy, name=resource.RelativeName(), updateMask=','.join(update_mask))
        return self.service.Patch(request)

    def Delete(self, resource):
        return self.service.Delete(self.messages.VmwareengineProjectsLocationsNetworkPoliciesDeleteRequest(name=resource.RelativeName()))

    def List(self, location_resource):
        location = location_resource.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsNetworkPoliciesListRequest(parent=location)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='networkPolicies')