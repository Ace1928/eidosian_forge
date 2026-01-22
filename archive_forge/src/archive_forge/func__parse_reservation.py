from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _parse_reservation(self, rsv):
    if rsv['resource_type'] == 'physical:host':
        for key in ['vcpus', 'memory_mb', 'disk_gb', 'affinity', 'amount']:
            rsv.pop(key)
    elif rsv['resource_type'] == 'virtual:instance':
        for key in ['hypervisor_properties', 'max', 'min', 'before_end']:
            rsv.pop(key)
    return rsv