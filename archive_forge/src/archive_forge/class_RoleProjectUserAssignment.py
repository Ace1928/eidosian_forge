from openstack import resource
class RoleProjectUserAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/projects/%(project_id)s/users/%(user_id)s/roles'
    allow_list = True
    name = resource.Body('name')
    links = resource.Body('links')
    project_id = resource.URI('project_id')
    user_id = resource.URI('user_id')