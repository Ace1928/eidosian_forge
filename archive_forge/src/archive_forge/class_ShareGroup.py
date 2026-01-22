from openstack import resource
class ShareGroup(resource.Resource):
    resource_key = 'share_group'
    resources_key = 'share_groups'
    base_path = '/share-groups'
    _query_mapping = resource.QueryParameters('share_group_id')
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_head = False
    availability_zone = resource.Body('availability_zone', type=str)
    consistent_snapshot_support = resource.Body('consistent_snapshot_support', type=str)
    created_at = resource.Body('created_at', type=str)
    description = resource.Body('description', type=str)
    project_id = resource.Body('project_id', type=str)
    share_group_snapshot_id = resource.Body('share_group_snapshot_id', type=str)
    share_group_type_id = resource.Body('share_group_type_id', type=str)
    share_network_id = resource.Body('share_network_id', type=str)
    share_types = resource.Body('share_types', type=list)
    status = resource.Body('status', type=str)