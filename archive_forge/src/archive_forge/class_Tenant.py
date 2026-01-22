from openstack import resource
class Tenant(resource.Resource):
    resource_key = 'tenant'
    resources_key = 'tenants'
    base_path = '/tenants'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    description = resource.Body('description')
    is_enabled = resource.Body('enabled', type=bool)
    name = resource.Body('name')