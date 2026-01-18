from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def update_default(module, array, current_default):
    """Update Default Protection"""
    changed = False
    current = []
    for default in range(0, len(current_default)):
        if module.params['scope'] == 'array':
            current.append(current_default[default].name)
        else:
            current.append(current_default[default].name.split(':')[-1])
    pg_list = []
    if module.params['state'] == 'present':
        if current:
            new_list = sorted(list(set(module.params['name'] + current)))
        else:
            new_list = sorted(list(set(module.params['name'])))
    elif current:
        new_list = sorted(list(set(current).difference(module.params['name'])))
    else:
        new_list = []
    if not new_list:
        delete_default(module, array)
    elif new_list == current:
        changed = False
    else:
        changed = True
        if not module.check_mode:
            for pgroup in range(0, len(new_list)):
                if module.params['scope'] == 'array':
                    pg_list.append(flasharray.DefaultProtectionReference(name=new_list[pgroup], type='protection_group'))
                else:
                    pg_list.append(flasharray.DefaultProtectionReference(name=module.params['pod'] + '::' + new_list[pgroup], type='protection_group'))
                if module.params['scope'] == 'array':
                    protection = flasharray.ContainerDefaultProtection(name='', type='', default_protections=pg_list)
                    res = array.patch_container_default_protections(names=[''], container_default_protection=protection)
                else:
                    protection = flasharray.ContainerDefaultProtection(name=module.params['pod'], type='pod', default_protections=pg_list)
                    res = array.patch_container_default_protections(names=[module.params['pod']], container_default_protection=protection)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to update default protection. Error: {0}'.format(res.errors[0].message))
    module.exit_json(changed=changed)