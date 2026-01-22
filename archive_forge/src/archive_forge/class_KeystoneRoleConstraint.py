from heat.common import exception
from heat.engine import constraints
class KeystoneRoleConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_role_id'
    entity = 'KeystoneRole'