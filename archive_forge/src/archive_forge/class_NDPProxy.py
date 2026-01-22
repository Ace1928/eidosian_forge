from openstack import resource
class NDPProxy(resource.Resource):
    resource_name = 'ndp proxy'
    resource_key = 'ndp_proxy'
    resources_key = 'ndp_proxies'
    base_path = '/ndp_proxies'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _allow_unknown_attrs_in_body = True
    _query_mapping = resource.QueryParameters('sort_key', 'sort_dir', 'name', 'description', 'project_id', 'router_id', 'port_id', 'ip_address')
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    id = resource.Body('id')
    ip_address = resource.Body('ip_address')
    name = resource.Body('name')
    port_id = resource.Body('port_id')
    project_id = resource.Body('project_id')
    revision_number = resource.Body('revision_number')
    router_id = resource.Body('router_id')
    updated_at = resource.Body('updated_at')