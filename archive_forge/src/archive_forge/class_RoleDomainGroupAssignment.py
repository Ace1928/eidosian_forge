from openstack import resource
class RoleDomainGroupAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/domains/%(domain_id)s/groups/%(group_id)s/roles'
    allow_list = True
    name = resource.Body('name')
    links = resource.Body('links')
    domain_id = resource.URI('domain_id')
    group_id = resource.URI('group_id')