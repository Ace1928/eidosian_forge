from neutron_lib.api.definitions import floatingip_pools
from neutron_lib.tests.unit.api.definitions import base
class FloatingIPPoolDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = floatingip_pools
    extension_resources = (floatingip_pools.FLOATINGIP_POOLS,)
    extension_attributes = ('subnet_id', 'subnet_name')