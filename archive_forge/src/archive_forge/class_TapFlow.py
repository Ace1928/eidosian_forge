from openstack import resource
class TapFlow(resource.Resource):
    """Tap Flow"""
    resource_key = 'tap_flow'
    resources_key = 'tap_flows'
    base_path = '/taas/tap_flows'
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
    tap_service_id = resource.Body('tap_service_id')
    direction = resource.Body('direction')
    status = resource.Body('status')
    source_port = resource.Body('source_port')