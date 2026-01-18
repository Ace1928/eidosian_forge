from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_cert(module, blade, cert):
    """Update certificate"""
    changed = False
    if cert.certificate_type == 'external':
        module.fail_json(msg='External certificates cannot be modified')
    if not module.params['private_key']:
        module.fail_json(msg='private_key must be specified for the global certificate')
    if cert.certificate.strip() != module.params['contents'].strip():
        changed = True
        if not module.check_mode:
            try:
                body = Certificate(certificate=module.params['contents'], private_key=module.params['private_key'])
                if module.params['passphrase']:
                    Certificate.passphrase = module.params['passphrase']
                blade.certificates.update_certificates(names=[module.params['name']], certificate=body)
            except Exception:
                module.fail_json(msg='Failed to create certificate {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)