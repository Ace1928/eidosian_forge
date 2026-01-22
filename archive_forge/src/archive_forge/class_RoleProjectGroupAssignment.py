from openstack import resource
class RoleProjectGroupAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/projects/%(project_id)s/groups/%(group_id)s/roles'
    allow_list = True
    name = resource.Body('name')
    links = resource.Body('links')
    project_id = resource.URI('project_id')
    group_id = resource.URI('group_id')