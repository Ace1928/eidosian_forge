from openstack import resource
class BgpVpnRouterAssociation(resource.Resource):
    resource_key = 'router_association'
    resources_key = 'router_associations'
    base_path = '/bgpvpn/bgpvpns/%(bgpvpn_id)s/router_associations'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    id = resource.Body('id')
    bgpvpn_id = resource.URI('bgpvpn_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    router_id = resource.Body('router_id')
    advertise_extra_routes = resource.Body('advertise_extra_routes')