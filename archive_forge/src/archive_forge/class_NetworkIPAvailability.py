from openstack import resource
class NetworkIPAvailability(resource.Resource):
    resource_key = 'network_ip_availability'
    resources_key = 'network_ip_availabilities'
    base_path = '/network-ip-availabilities'
    name_attribute = 'network_name'
    _allow_unknown_attrs_in_body = True
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True
    _query_mapping = resource.QueryParameters('ip_version', 'network_id', 'network_name', 'project_id', 'sort_key', 'sort_dir')
    network_id = resource.Body('network_id')
    network_name = resource.Body('network_name')
    subnet_ip_availability = resource.Body('subnet_ip_availability', type=list)
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    total_ips = resource.Body('total_ips', type=int)
    used_ips = resource.Body('used_ips', type=int)