from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _ensure_members(self, operation):
    if operation not in ['add', 'remove']:
        self.module.fail_json(msg='Bad operation: %s' % operation)
    rule = self.get_rule()
    if not rule:
        self.module.fail_json(msg='Unknown rule: %s' % self.module.params.get('name'))
    existing = {}
    for vm in self._get_members_of_rule(rule=rule):
        existing[vm['name']] = vm['id']
    wanted_names = self.module.params.get('vms')
    if operation == 'add':
        cs_func = 'assignToLoadBalancerRule'
        to_change = set(wanted_names) - set(existing.keys())
    else:
        cs_func = 'removeFromLoadBalancerRule'
        to_change = set(wanted_names) & set(existing.keys())
    if not to_change:
        return rule
    args = self._get_common_args()
    args['fetch_list'] = True
    vms = self.query_api('listVirtualMachines', **args)
    to_change_ids = []
    for name in to_change:
        for vm in vms:
            if vm['name'] == name:
                to_change_ids.append(vm['id'])
                break
        else:
            self.module.fail_json(msg='Unknown VM: %s' % name)
    if to_change_ids:
        self.result['changed'] = True
    if to_change_ids and (not self.module.check_mode):
        res = self.query_api(cs_func, id=rule['id'], virtualmachineids=to_change_ids)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            self.poll_job(res)
            rule = self.get_rule()
    return rule