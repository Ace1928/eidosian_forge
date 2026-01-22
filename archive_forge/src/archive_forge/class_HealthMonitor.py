from openstack import resource
class HealthMonitor(resource.Resource):
    resource_key = 'healthmonitor'
    resources_key = 'healthmonitors'
    base_path = '/lbaas/healthmonitors'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('delay', 'expected_codes', 'http_method', 'max_retries', 'timeout', 'type', 'url_path', 'project_id', is_admin_state_up='adminstate_up')
    delay = resource.Body('delay')
    expected_codes = resource.Body('expected_codes')
    http_method = resource.Body('http_method')
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    max_retries = resource.Body('max_retries')
    name = resource.Body('name')
    pool_ids = resource.Body('pools', type=list)
    pool_id = resource.Body('pool_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    timeout = resource.Body('timeout')
    type = resource.Body('type')
    url_path = resource.Body('url_path')