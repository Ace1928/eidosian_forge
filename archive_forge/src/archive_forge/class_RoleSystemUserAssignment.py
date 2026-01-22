from openstack import resource
class RoleSystemUserAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/system/users/%(user_id)s/roles'
    allow_list = True
    system_id = resource.URI('system_id')
    user_id = resource.URI('user_id')