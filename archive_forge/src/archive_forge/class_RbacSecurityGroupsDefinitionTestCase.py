from neutron_lib.api.definitions import rbac_security_groups
from neutron_lib.tests.unit.api.definitions import base
class RbacSecurityGroupsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = rbac_security_groups