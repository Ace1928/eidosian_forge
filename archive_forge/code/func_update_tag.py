from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_tag(module, array, current_tags):
    """Update Volume Tag"""
    changed = False
    for tag in range(0, len(module.params['kvp'])):
        tag_exists = False
        for current_tag in range(0, len(current_tags)):
            if module.params['kvp'][tag].split(':')[0] == current_tags[current_tag]['key'] and module.params['namespace'] == current_tags[current_tag]['namespace']:
                tag_exists = True
                if module.params['kvp'][tag].split(':')[1] != current_tags[current_tag]['value']:
                    changed = True
                    if not module.check_mode:
                        try:
                            array.add_tag_to_volume(module.params['name'], namespace=module.params['namespace'], key=module.params['kvp'][tag].split(':')[0], value=module.params['kvp'][tag].split(':')[1])
                        except Exception:
                            module.fail_json(msg="Failed to update tag '{0}' from volume {1}".format(module.params['kvp'][tag].split(':')[0], module.params['name']))
        if not tag_exists:
            changed = True
            if not module.check_mode:
                try:
                    array.add_tag_to_volume(module.params['name'], namespace=module.params['namespace'], key=module.params['kvp'][tag].split(':')[0], value=module.params['kvp'][tag].split(':')[1])
                except Exception:
                    module.fail_json(msg='Failed to add tag KVP {0} to volume {1}'.format(module.params['kvp'][tag].split(':')[0], module.params['name']))
    module.exit_json(changed=changed)