from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_nfs_policy(module, blade):
    """Update NFS Export Policy Rule"""
    changed = False
    if module.params['client']:
        current_policy_rule = blade.get_nfs_export_policies_rules(policy_names=[module.params['name']], filter="client='" + module.params['client'] + "'")
        if current_policy_rule.status_code == 200 and current_policy_rule.total_item_count == 0:
            rule = NfsExportPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], anonuid=module.params['anonuid'], anongid=module.params['anongid'], fileid_32bit=module.params['fileid_32bit'], atime=module.params['atime'], secure=module.params['secure'], security=module.params['security'])
            changed = True
            if not module.check_mode:
                if module.params['before_rule']:
                    before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                    res = blade.post_nfs_export_policies_rules(policy_names=[module.params['name']], rule=rule, before_rule_name=before_name)
                else:
                    res = blade.post_nfs_export_policies_rules(policy_names=[module.params['name']], rule=rule)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to create rule for client {0} in export policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
        else:
            rules = list(current_policy_rule.items)
            cli_count = None
            done = False
            if module.params['client'] == '*':
                for cli in range(0, len(rules)):
                    if rules[cli].client == '*':
                        cli_count = cli
                if not cli_count:
                    rule = NfsExportPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], anonuid=module.params['anonuid'], anongid=module.params['anongid'], fileid_32bit=module.params['fileid_32bit'], atime=module.params['atime'], secure=module.params['secure'], security=module.params['security'])
                    done = True
                    changed = True
                    if not module.check_mode:
                        if module.params['before_rule']:
                            res = blade.post_nfs_export_policies_rules(policy_names=[module.params['name']], rule=rule, before_rule_name=(module.params['name'] + '.' + str(module.params['before_rule']),))
                        else:
                            res = blade.post_nfs_export_policies_rules(policy_names=[module.params['name']], rule=rule)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to create rule for client {0} in export policy {1}. Error: {2}'.format(module.params['client'], module.params['name'], res.errors[0].message))
            if not done:
                old_policy_rule = rules[0]
                current_rule = {'anongid': getattr(old_policy_rule, 'anongid', None), 'anonuid': getattr(old_policy_rule, 'anonuid', None), 'atime': old_policy_rule.atime, 'client': sorted(old_policy_rule.client), 'fileid_32bit': old_policy_rule.fileid_32bit, 'permission': sorted(old_policy_rule.permission), 'secure': old_policy_rule.secure, 'security': sorted(old_policy_rule.security)}
                if module.params['permission']:
                    new_permission = sorted(module.params['permission'])
                else:
                    new_permission = sorted(current_rule['permission'])
                if module.params['client']:
                    new_client = sorted(module.params['client'])
                else:
                    new_client = sorted(current_rule['client'])
                if module.params['security']:
                    new_security = sorted(module.params['security'])
                else:
                    new_security = sorted(current_rule['security'])
                if module.params['anongid']:
                    new_anongid = module.params['anongid']
                else:
                    new_anongid = current_rule['anongid']
                if module.params['anonuid']:
                    new_anonuid = module.params['anonuid']
                else:
                    new_anonuid = current_rule['anonuid']
                if module.params['atime'] != current_rule['atime']:
                    new_atime = module.params['atime']
                else:
                    new_atime = current_rule['atime']
                if module.params['secure'] != current_rule['secure']:
                    new_secure = module.params['secure']
                else:
                    new_secure = current_rule['secure']
                if module.params['fileid_32bit'] != current_rule['fileid_32bit']:
                    new_fileid_32bit = module.params['fileid_32bit']
                else:
                    new_fileid_32bit = current_rule['fileid_32bit']
                new_rule = {'anongid': new_anongid, 'anonuid': new_anonuid, 'atime': new_atime, 'client': new_client, 'fileid_32bit': new_fileid_32bit, 'permission': new_permission, 'secure': new_secure, 'security': new_security}
                if current_rule != new_rule:
                    changed = True
                    if not module.check_mode:
                        rule = NfsExportPolicyRule(client=module.params['client'], permission=module.params['permission'], access=module.params['access'], anonuid=module.params['anonuid'], anongid=module.params['anongid'], fileid_32bit=module.params['fileid_32bit'], atime=module.params['atime'], secure=module.params['secure'], security=module.params['security'])
                        res = blade.patch_nfs_export_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=rule)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to update NFS export rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
                if module.params['before_rule'] and module.params['before_rule'] != old_policy_rule.index:
                    changed = True
                    if not module.check_mode:
                        before_name = module.params['name'] + '.' + str(module.params['before_rule'])
                        res = blade.patch_nfs_export_policies_rules(names=[module.params['name'] + '.' + str(old_policy_rule.index)], rule=NfsExportPolicyRule(), before_rule_name=before_name)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to move NFS export rule {0}. Error: {1}'.format(module.params['name'] + '.' + str(old_policy_rule.index), res.errors[0].message))
    current_policy = list(blade.get_nfs_export_policies(names=[module.params['name']]).items)[0]
    if current_policy.enabled != module.params['enabled']:
        changed = True
        if not module.check_mode:
            res = blade.patch_nfs_export_policies(policy=NfsExportPolicy(enabled=module.params['enabled']), names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to change state of nfs export policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)