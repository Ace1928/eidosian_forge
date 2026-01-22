from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _update_host_initiators(module, array, answer=False):
    """Change host initiator if iscsi or nvme or add new FC WWNs"""
    if module.params['nqn']:
        current_nqn = array.get_host(module.params['name'])['nqn']
        if module.params['nqn'] != ['']:
            if current_nqn != module.params['nqn']:
                answer = True
                if not module.check_mode:
                    try:
                        array.set_host(module.params['name'], nqnlist=module.params['nqn'])
                    except Exception:
                        module.fail_json(msg='Change of NVMe NQN failed.')
        elif current_nqn:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], remnqnlist=current_nqn)
                except Exception:
                    module.fail_json(msg='Removal of NVMe NQN failed.')
    if module.params['iqn']:
        current_iqn = array.get_host(module.params['name'])['iqn']
        if module.params['iqn'] != ['']:
            if current_iqn != module.params['iqn']:
                answer = True
                if not module.check_mode:
                    try:
                        array.set_host(module.params['name'], iqnlist=module.params['iqn'])
                    except Exception:
                        module.fail_json(msg='Change of iSCSI IQN failed.')
        elif current_iqn:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], remiqnlist=current_iqn)
                except Exception:
                    module.fail_json(msg='Removal of iSCSI IQN failed.')
    if module.params['wwns']:
        module.params['wwns'] = [wwn.replace(':', '') for wwn in module.params['wwns']]
        module.params['wwns'] = [wwn.upper() for wwn in module.params['wwns']]
        current_wwn = array.get_host(module.params['name'])['wwn']
        if module.params['wwns'] != ['']:
            if current_wwn != module.params['wwns']:
                answer = True
                if not module.check_mode:
                    try:
                        array.set_host(module.params['name'], wwnlist=module.params['wwns'])
                    except Exception:
                        module.fail_json(msg='FC WWN change failed.')
        elif current_wwn:
            answer = True
            if not module.check_mode:
                try:
                    array.set_host(module.params['name'], remwwnlist=current_wwn)
                except Exception:
                    module.fail_json(msg='Removal of all FC WWNs failed.')
    return answer