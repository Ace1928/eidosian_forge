from openstack import resource
class BgpVpnNetworkAssociation(resource.Resource):
    resource_key = 'network_association'
    resources_key = 'network_associations'
    base_path = '/bgpvpn/bgpvpns/%(bgpvpn_id)s/network_associations'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    id = resource.Body('id')
    bgpvpn_id = resource.URI('bgpvpn_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    network_id = resource.Body('network_id')