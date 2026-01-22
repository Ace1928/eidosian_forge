from heat.common import exception
from heat.engine import constraints
class KeystoneRegionConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_region_id'
    entity = 'KeystoneRegion'