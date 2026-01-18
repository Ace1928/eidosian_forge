from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def update_iso(self, iso):
    args = self._get_common_args()
    args.update({'id': iso['id']})
    if self.has_changed(args, iso):
        self.result['changed'] = True
        if not self.module.params.get('cross_zones'):
            args['zoneid'] = self.get_zone(key='id')
        else:
            self.result['cross_zones'] = True
            args['zoneid'] = -1
        if not self.module.check_mode:
            res = self.query_api('updateIso', **args)
            self.iso = res['iso']
    return self.iso