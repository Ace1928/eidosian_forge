from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_certgrp(module, blade):
    """Update certificate group"""
    changed = False
    try:
        certs = blade.certificate_groups.list_certificate_group_certificates(certificate_group_names=[module.params['name']])
    except Exception:
        module.fail_json(msg='Failed to get certifates list for group {0}.'.format(module.params['name']))
    if not certs:
        if module.params['state'] == 'present':
            changed = True
            if not module.check_mode:
                try:
                    blade.certificate_groups.add_certificate_group_certificates(certificate_names=module.params['certificates'], certificate_group_names=[module.params['name']])
                except Exception:
                    module.fail_json(msg='Failed to add certifcates {0}. Please check they all exist'.format(module.params['certificates']))
    else:
        current = []
        for cert in range(0, len(certs.items)):
            current.append(certs.items[cert].member.name)
        for new_cert in range(0, len(module.params['certificates'])):
            certificate = module.params['certificates'][new_cert]
            if certificate in current:
                if module.params['state'] == 'absent':
                    changed = True
                    if not module.check_mode:
                        try:
                            blade.certificate_groups.remove_certificate_group_certificates(certificate_names=[certificate], certificate_group_names=[module.params['name']])
                        except Exception:
                            module.fail_json(msg='Failed to delete certifcate {0} from group {1}.'.format(certificate, module.params['name']))
            elif module.params['state'] == 'present':
                changed = True
                if not module.check_mode:
                    try:
                        blade.certificate_groups.add_certificate_group_certificates(certificate_names=[certificate], certificate_group_names=[module.params['name']])
                    except Exception:
                        module.fail_json(msg='Failed to add certifcate {0} to group {1}'.format(certificate, module.params['name']))
    module.exit_json(changed=changed)