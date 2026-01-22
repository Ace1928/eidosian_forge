from openstack import resource
class DeviceProfile(resource.Resource):
    resource_key = 'device_profile'
    resources_key = 'device_profiles'
    base_path = '/device_profiles'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    groups = resource.Body('groups')
    name = resource.Body('name')
    updated_at = resource.Body('updated_at')
    uuid = resource.Body('uuid', alternate_id=True)

    def _prepare_request_body(self, patch, prepend_key):
        body = super(DeviceProfile, self)._prepare_request_body(patch, prepend_key)
        return [body]

    def create(self, session, base_path=None):
        return super(DeviceProfile, self).create(session, prepend_key=False, base_path=base_path)