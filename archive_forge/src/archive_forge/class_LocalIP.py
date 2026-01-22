from openstack import resource
class LocalIP(resource.Resource):
    """Local IP extension."""
    resource_name = 'local ip'
    resource_key = 'local_ip'
    resources_key = 'local_ips'
    base_path = '/local_ips'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _allow_unknown_attrs_in_body = True
    _query_mapping = resource.QueryParameters('sort_key', 'sort_dir', 'name', 'description', 'project_id', 'network_id', 'local_port_id', 'local_ip_address', 'ip_mode')
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    id = resource.Body('id')
    ip_mode = resource.Body('ip_mode')
    local_ip_address = resource.Body('local_ip_address')
    local_port_id = resource.Body('local_port_id')
    name = resource.Body('name')
    network_id = resource.Body('network_id')
    project_id = resource.Body('project_id')
    revision_number = resource.Body('revision_number')
    updated_at = resource.Body('updated_at')