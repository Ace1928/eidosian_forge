from openstack import resource
class ShareSnapshot(resource.Resource):
    resource_key = 'snapshot'
    resources_key = 'snapshots'
    base_path = '/snapshots'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('snapshot_id')
    created_at = resource.Body('created_at')
    description = resource.Body('description', type=str)
    display_name = resource.Body('display_name', type=str)
    display_description = resource.Body('display_description', type=str)
    project_id = resource.Body('project_id', type=str)
    share_id = resource.Body('share_id', type=str)
    share_proto = resource.Body('share_proto', type=str)
    share_size = resource.Body('share_size', type=int)
    size = resource.Body('size', type=int)
    status = resource.Body('status', type=str)
    user_id = resource.Body('user_id', type=str)