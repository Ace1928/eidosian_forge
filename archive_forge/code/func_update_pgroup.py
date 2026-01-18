from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_pgroup(module, array):
    """Update Protection Group"""
    changed = renamed = False
    api_version = array._list_available_rest_versions()
    if module.params['target']:
        connected_targets = []
        connected_arrays = get_arrays(array)
        if OFFLOAD_API_VERSION in api_version:
            connected_targets = get_targets(array)
        connected_arrays = connected_arrays + connected_targets
        if connected_arrays == []:
            module.fail_json(msg='No targets connected to source array.')
        current_connects = array.get_pgroup(module.params['name'])['targets']
        current_targets = []
        if current_connects:
            for targetcnt in range(0, len(current_connects)):
                current_targets.append(current_connects[targetcnt]['name'])
        if set(module.params['target'][0:4]) != set(current_targets):
            if not set(module.params['target'][0:4]).issubset(connected_arrays):
                module.fail_json(msg='Check all selected targets are connected to the source array.')
            changed = True
            if not module.check_mode:
                try:
                    array.set_pgroup(module.params['name'], targetlist=module.params['target'][0:4])
                except Exception:
                    module.fail_json(msg='Changing targets for pgroup {0} failed.'.format(module.params['name']))
    if module.params['target'] and module.params['enabled'] != get_pgroup_sched(module, array)['replicate_enabled']:
        changed = True
        if not module.check_mode:
            try:
                array.set_pgroup(module.params['name'], replicate_enabled=module.params['enabled'])
            except Exception:
                module.fail_json(msg='Changing enabled status of pgroup {0} failed.'.format(module.params['name']))
    elif not module.params['target'] and module.params['enabled'] != get_pgroup_sched(module, array)['snap_enabled']:
        changed = True
        if not module.check_mode:
            try:
                array.set_pgroup(module.params['name'], snap_enabled=module.params['enabled'])
            except Exception:
                module.fail_json(msg='Changing enabled status of pgroup {0} failed.'.format(module.params['name']))
    if module.params['volume'] and get_pgroup(module, array)['hosts'] is None and (get_pgroup(module, array)['hgroups'] is None):
        if get_pgroup(module, array)['volumes'] is None:
            if not module.check_mode:
                changed = True
                try:
                    array.set_pgroup(module.params['name'], vollist=module.params['volume'])
                except Exception:
                    module.fail_json(msg='Adding volumes to pgroup {0} failed.'.format(module.params['name']))
        else:
            cased_vols = list(module.params['volume'])
            cased_pgvols = list(get_pgroup(module, array)['volumes'])
            if not all((x in cased_pgvols for x in cased_vols)):
                if not module.check_mode:
                    changed = True
                    try:
                        array.set_pgroup(module.params['name'], addvollist=module.params['volume'])
                    except Exception:
                        module.fail_json(msg='Changing volumes in pgroup {0} failed.'.format(module.params['name']))
    if module.params['host'] and get_pgroup(module, array)['volumes'] is None and (get_pgroup(module, array)['hgroups'] is None):
        if get_pgroup(module, array)['hosts'] is None:
            if not module.check_mode:
                changed = True
                try:
                    array.set_pgroup(module.params['name'], hostlist=module.params['host'])
                except Exception:
                    module.fail_json(msg='Adding hosts to pgroup {0} failed.'.format(module.params['name']))
        else:
            cased_hosts = list(module.params['host'])
            cased_pghosts = list(get_pgroup(module, array)['hosts'])
            if not all((x in cased_pghosts for x in cased_hosts)):
                if not module.check_mode:
                    changed = True
                    try:
                        array.set_pgroup(module.params['name'], addhostlist=module.params['host'])
                    except Exception:
                        module.fail_json(msg='Changing hosts in pgroup {0} failed.'.format(module.params['name']))
    if module.params['hostgroup'] and get_pgroup(module, array)['hosts'] is None and (get_pgroup(module, array)['volumes'] is None):
        if get_pgroup(module, array)['hgroups'] is None:
            if not module.check_mode:
                changed = True
                try:
                    array.set_pgroup(module.params['name'], hgrouplist=module.params['hostgroup'])
                except Exception:
                    module.fail_json(msg='Adding hostgroups to pgroup {0} failed.'.format(module.params['name']))
        else:
            cased_hostg = list(module.params['hostgroup'])
            cased_pghostg = list(get_pgroup(module, array)['hgroups'])
            if not all((x in cased_pghostg for x in cased_hostg)):
                if not module.check_mode:
                    changed = True
                    try:
                        array.set_pgroup(module.params['name'], addhgrouplist=module.params['hostgroup'])
                    except Exception:
                        module.fail_json(msg='Changing hostgroups in pgroup {0} failed.'.format(module.params['name']))
    if module.params['rename']:
        if not rename_exists(module, array):
            if ':' in module.params['name']:
                container = module.params['name'].split(':')[0]
                if '::' in module.params['name']:
                    rename = container + '::' + module.params['rename']
                else:
                    rename = container + ':' + module.params['rename']
            else:
                rename = module.params['rename']
            renamed = True
            if not module.check_mode:
                try:
                    array.rename_pgroup(module.params['name'], rename)
                    module.params['name'] = rename
                except Exception:
                    module.fail_json(msg='Rename to {0} failed.'.format(rename))
        else:
            module.warn('Rename failed. Protection group {0} already exists in container. Continuing with other changes...'.format(module.params['rename']))
    if RETENTION_LOCK_VERSION in api_version:
        arrayv6 = get_array(module)
        current_pg = list(arrayv6.get_protection_groups(names=[module.params['name']]).items)[0]
        if current_pg.retention_lock == 'unlocked' and module.params['safe_mode']:
            changed = True
            if not module.check_mode:
                res = arrayv6.patch_protection_groups(names=[module.params['name']], protection_group=flasharray.ProtectionGroup(retention_lock='ratcheted'))
                if res.status_code != 200:
                    module.fail_json(msg='Failed to set SafeMode on protection group {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
        if current_pg.retention_lock == 'ratcheted' and (not module.params['safe_mode']):
            module.warn('Disabling SafeMode on protection group {0} can only be performed by Pure Technical Support'.format(module.params['name']))
    changed = changed or renamed
    module.exit_json(changed=changed)