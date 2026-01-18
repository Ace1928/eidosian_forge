from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def update_rule(module, blade, rule, bladev2=None):
    """Update snapshot policy"""
    changed = False
    if not bladev2:
        current_rule = {'prefix': rule.prefix, 'keep_previous_version_for': rule.keep_previous_version_for, 'enabled': rule.enabled}
    else:
        current_rule = {'prefix': rule.prefix, 'abort_incomplete_multipart_uploads_after': rule.abort_incomplete_multipart_uploads_after, 'keep_current_version_for': rule.keep_current_version_for, 'keep_previous_version_for': rule.keep_previous_version_for, 'keep_current_version_until': rule.keep_current_version_until, 'enabled': rule.enabled}
    if not module.params['prefix']:
        prefix = current_rule['prefix']
    else:
        prefix = module.params['prefix']
    if not module.params['keep_previous_for']:
        keep_previous_for = current_rule['keep_previous_version_for']
    else:
        keep_previous_for = _convert_to_millisecs(module.params['keep_previous_for'])
    if bladev2:
        if not module.params['keep_current_for']:
            keep_current_for = current_rule['keep_current_version_for']
        else:
            keep_current_for = _convert_to_millisecs(module.params['keep_current_for'])
        if not module.params['abort_uploads_after']:
            abort_uploads_after = current_rule['abort_incomplete_multipart_uploads_after']
        else:
            abort_uploads_after = _convert_to_millisecs(module.params['abort_uploads_after'])
        if not module.params['keep_current_until']:
            keep_current_until = current_rule['keep_current_version_until']
        else:
            keep_current_until = module.params['keep_current_until']
        new_rule = {'prefix': prefix, 'abort_incomplete_multipart_uploads_after': abort_uploads_after, 'keep_current_version_for': keep_current_for, 'keep_previous_version_for': keep_previous_for, 'keep_current_version_until': keep_current_until, 'enabled': module.params['enabled']}
    else:
        new_rule = {'prefix': prefix, 'keep_previous_version_for': keep_previous_for, 'enabled': module.params['enabled']}
    if current_rule != new_rule:
        changed = True
        if not module.check_mode:
            if not bladev2:
                try:
                    attr = LifecycleRulePatch(keep_previous_version_for=new_rule['keep_previous_version_for'], prefix=new_rule['prefix'])
                    attr.enabled = module.params['enabled']
                    blade.lifecycle_rules.update_lifecycle_rules(names=[module.params['bucket'] + '/' + module.params['name']], rule=attr, confirm_date=True)
                except Exception:
                    module.fail_json(msg='Failed to update lifecycle rule {0} for bucket {1}.'.format(module.params['name'], module.params['bucket']))
            else:
                attr = flashblade.LifecycleRulePatch(keep_previous_version_for=new_rule['keep_previous_version_for'], keep_current_version_for=new_rule['keep_current_version_for'], keep_current_version_until=new_rule['keep_current_version_until'], abort_incomplete_multipart_uploads_after=new_rule['abort_incomplete_multipart_uploads_after'], prefix=new_rule['prefix'], enabled=new_rule['enabled'])
                if attr.keep_current_version_until:
                    res = bladev2.patch_lifecycle_rules(names=[module.params['bucket'] + '/' + module.params['name']], lifecycle=attr, confirm_date=True)
                else:
                    res = bladev2.patch_lifecycle_rules(names=[module.params['bucket'] + '/' + module.params['name']], lifecycle=attr)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to update lifecycle rule {0} for bucket {1}. Error: {2}'.format(module.params['name'], module.params['bucket'], res.errors[0].message))
    module.exit_json(changed=changed)