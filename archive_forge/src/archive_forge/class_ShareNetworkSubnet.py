from openstack import resource
class ShareNetworkSubnet(resource.Resource):
    resource_key = 'share_network_subnet'
    resources_key = 'share_network_subnets'
    base_path = '/share-networks/%(share_network_id)s/subnets'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    share_network_id = resource.URI('share_network_id', type=str)
    availability_zone = resource.Body('availability_zone', type=str)
    cidr = resource.Body('cidr', type=str)
    created_at = resource.Body('created_at')
    gateway = resource.Body('gateway', type=str)
    ip_version = resource.Body('ip_version', type=int)
    mtu = resource.Body('mtu', type=str)
    network_type = resource.Body('network_type', type=str)
    neutron_net_id = resource.Body('neutron_net_id', type=str)
    neutron_subnet_id = resource.Body('neutron_subnet_id', type=str)
    segmentation_id = resource.Body('segmentation_id', type=int)
    share_network_name = resource.Body('share_network_name', type=str)
    updated_at = resource.Body('updated_at', type=str)

    def create(self, session, **kwargs):
        return super().create(session, resource_request_key='share-network-subnet', **kwargs)