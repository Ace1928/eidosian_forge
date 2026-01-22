from openstack import resource
class ResourceLock(resource.Resource):
    resource_key = 'resource_lock'
    resources_key = 'resource_locks'
    base_path = '/resource-locks'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('project_id', 'created_since', 'created_before', 'limit', 'offset', 'id', 'resource_id', 'resource_type', 'resource_action', 'user_id', 'lock_context', 'lock_reason', 'lock_reason~', 'sort_key', 'sort_dir', 'with_count', 'all_projects')
    _max_microversion = '2.81'
    created_at = resource.Body('created_at', type=str)
    updated_at = resource.Body('updated_at', type=str)
    user_id = resource.Body('user_id', type=str)
    project_id = resource.Body('project_id', type=str)
    resource_type = resource.Body('resource_type', type=str)
    resource_id = resource.Body('resource_id', type=str)
    resource_action = resource.Body('resource_action', type=str)
    lock_reason = resource.Body('lock_reason', type=str)
    lock_context = resource.Body('lock_context', type=str)