from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def update_rule_spec(self, rule_obj=None):
    """
        Update DRS rule
        """
    changed = False
    result = None
    rule_obj.vm = self.vm_obj_list
    if rule_obj.mandatory != self.mandatory:
        rule_obj.mandatory = self.mandatory
    if rule_obj.enabled != self.enabled:
        rule_obj.enabled = self.enabled
    rule_spec = vim.cluster.RuleSpec(info=rule_obj, operation='edit')
    config_spec = vim.cluster.ConfigSpec(rulesSpec=[rule_spec])
    try:
        if not self.module.check_mode:
            task = self.cluster_obj.ReconfigureCluster_Task(config_spec, modify=True)
            changed, result = wait_for_task(task)
        else:
            changed = True
    except vmodl.fault.InvalidRequest as e:
        result = to_native(e.msg)
    except Exception as e:
        result = to_native(e)
    if changed:
        rule_obj = self.get_rule_key_by_name(rule_name=self.rule_name)
        result = self.normalize_rule_spec(rule_obj)
    return (changed, result)