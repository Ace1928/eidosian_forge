from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_zone(self):
    zone = self.get_zone()
    if zone:
        self.result['changed'] = True
        args = {'id': zone['id']}
        if not self.module.check_mode:
            self.query_api('deleteZone', **args)
    return zone