from openstack import resource
class RegisteredLimit(resource.Resource):
    resource_key = 'registered_limit'
    resources_key = 'registered_limits'
    base_path = '/registered_limits'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters('service_id', 'region_id', 'resource_name')
    description = resource.Body('description')
    links = resource.Body('links')
    service_id = resource.Body('service_id')
    region_id = resource.Body('region_id')
    resource_name = resource.Body('resource_name')
    default_limit = resource.Body('default_limit')

    def _prepare_request_body(self, patch, prepend_key):
        body = self._body.dirty
        if prepend_key and self.resource_key is not None:
            if patch:
                body = {self.resource_key: body}
            else:
                body = {self.resources_key: [body]}
        return body