from openstack import resource
class VpnService(resource.Resource):
    resource_key = 'vpnservice'
    resources_key = 'vpnservices'
    base_path = '/vpn/vpnservices'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'external_v4_ip', 'external_v6_ip', 'name', 'router_id', 'project_id', 'tenant_id', 'subnet_id', is_admin_state_up='admin_state_up')
    description = resource.Body('description')
    external_v4_ip = resource.Body('external_v4_ip')
    external_v6_ip = resource.Body('external_v6_ip')
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    name = resource.Body('name')
    router_id = resource.Body('router_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    status = resource.Body('status')
    subnet_id = resource.Body('subnet_id')