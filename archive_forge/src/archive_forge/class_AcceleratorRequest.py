from openstack import exceptions
from openstack import resource
class AcceleratorRequest(resource.Resource):
    resource_key = 'arq'
    resources_key = 'arqs'
    base_path = '/accelerator_requests'
    allow_create = True
    allow_fetch = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    attach_handle_info = resource.Body('attach_handle_info')
    attach_handle_type = resource.Body('attach_handle_type')
    device_profile_name = resource.Body('device_profile_name')
    device_profile_group_id = resource.Body('device_profile_group_id')
    device_rp_uuid = resource.Body('device_rp_uuid')
    hostname = resource.Body('hostname')
    instance_uuid = resource.Body('instance_uuid')
    state = resource.Body('state')
    uuid = resource.Body('uuid', alternate_id=True)

    def _convert_patch(self, patch):
        converted = super(AcceleratorRequest, self)._convert_patch(patch)
        converted = {self.id: converted}
        return converted

    def patch(self, session, patch=None, prepend_key=True, has_body=True, retry_on_conflict=None, base_path=None):
        self._body._dirty.discard('id')
        if not patch and (not self.requires_commit):
            return self
        if not self.allow_patch:
            raise exceptions.MethodNotSupported(self, 'patch')
        request = self._prepare_request(prepend_key=prepend_key, base_path=base_path, patch=True)
        microversion = self._get_microversion(session, action='patch')
        if patch:
            request.body = self._convert_patch(patch)
        return self._commit(session, request, 'PATCH', microversion, has_body=has_body, retry_on_conflict=retry_on_conflict)

    def _consume_attrs(self, mapping, attrs):
        if isinstance(self, AcceleratorRequest):
            if self.resources_key in attrs:
                attrs = attrs[self.resources_key][0]
        return super(AcceleratorRequest, self)._consume_attrs(mapping, attrs)

    def create(self, session, base_path=None):
        return super(AcceleratorRequest, self).create(session, prepend_key=False, base_path=base_path)