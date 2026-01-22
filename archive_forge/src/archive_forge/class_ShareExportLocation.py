from openstack import resource
class ShareExportLocation(resource.Resource):
    resource_key = 'export_location'
    resources_key = 'export_locations'
    base_path = '/shares/%(share_id)s/export_locations'
    allow_list = True
    allow_fetch = True
    allow_create = False
    allow_commit = False
    allow_delete = False
    allow_head = False
    _max_microversion = '2.47'
    share_id = resource.URI('share_id', type='str')
    path = resource.Body('path', type=str)
    is_preferred = resource.Body('preferred', type=bool)
    share_instance_id = resource.Body('share_instance_id', type=str)
    is_admin = resource.Body('is_admin_only', type=bool)
    created_at = resource.Body('created_at', type=str)
    updated_at = resource.Body('updated_at', type=str)