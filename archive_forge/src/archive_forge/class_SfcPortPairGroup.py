from openstack import resource
class SfcPortPairGroup(resource.Resource):
    resource_key = 'port_pair_group'
    resources_key = 'port_pair_groups'
    base_path = '/sfc/port_pair_groups'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'tenant_id')
    description = resource.Body('description')
    name = resource.Body('name')
    port_pairs = resource.Body('port_pairs', type=list)
    port_pair_group_parameters = resource.Body('port_pair_group_parameters', type=dict)
    is_tap_enabled = resource.Body('tap_enabled', type=bool)
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)