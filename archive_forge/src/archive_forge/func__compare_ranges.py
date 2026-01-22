from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _compare_ranges(self, entity):
    if self._module.params['ranges'] is not None:
        ranges = sorted(['%s,%s' % (mac_range.from_, mac_range.to) for mac_range in entity.ranges])
        return equal(sorted(self._module.params['ranges']), ranges)
    return True