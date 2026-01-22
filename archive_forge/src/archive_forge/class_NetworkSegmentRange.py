from openstack import resource
class NetworkSegmentRange(resource.Resource):
    resource_key = 'network_segment_range'
    resources_key = 'network_segment_ranges'
    base_path = '/network_segment_ranges'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'default', 'shared', 'project_id', 'network_type', 'physical_network', 'minimum', 'maximum', 'used', 'available', 'sort_key', 'sort_dir')
    name = resource.Body('name')
    default = resource.Body('default', type=bool)
    shared = resource.Body('shared', type=bool)
    project_id = resource.Body('project_id')
    network_type = resource.Body('network_type')
    physical_network = resource.Body('physical_network')
    minimum = resource.Body('minimum', type=int)
    maximum = resource.Body('maximum', type=int)
    used = resource.Body('used', type=dict)
    available = resource.Body('available', type=list)