from openstack import resource
class TapService(resource.Resource):
    """Tap Service"""
    resource_key = 'tap_service'
    resources_key = 'tap_services'
    base_path = '/taas/tap_services'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _allow_unknown_attrs_in_body = True
    _query_mapping = resource.QueryParameters('sort_key', 'sort_dir', 'name', 'project_id')
    id = resource.Body('id')
    name = resource.Body('name')
    description = resource.Body('description')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    port_id = resource.Body('port_id')
    status = resource.Body('status')