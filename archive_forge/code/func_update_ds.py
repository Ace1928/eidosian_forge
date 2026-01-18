from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_ds(module, blade):
    """Update Directory Service"""
    mod_ds = False
    changed = False
    password_required = False
    attr = {}
    try:
        ds_now = blade.directory_services.list_directory_services(names=[module.params['dstype']]).items[0]
        if module.params['dstype'] == 'nfs' and module.params['nis_servers']:
            if sorted(module.params['nis_servers']) != sorted(ds_now.nfs.nis_servers) or module.params['nis_domain'] != ''.join(map(str, ds_now.nfs.nis_domains)):
                attr['nfs'] = {'nis_domains': [module.params['nis_domain']], 'nis_servers': module.params['nis_servers'][0:30]}
                mod_ds = True
        else:
            if module.params['uri']:
                if sorted(module.params['uri'][0:30]) != sorted(ds_now.uris):
                    attr['uris'] = module.params['uri'][0:30]
                    mod_ds = True
                    password_required = True
            if module.params['base_dn']:
                if module.params['base_dn'] != ds_now.base_dn:
                    attr['base_dn'] = module.params['base_dn']
                    mod_ds = True
            if module.params['bind_user']:
                if module.params['bind_user'] != ds_now.bind_user:
                    password_required = True
                    attr['bind_user'] = module.params['bind_user']
                    mod_ds = True
                elif module.params['force_bind_password']:
                    password_required = True
                    mod_ds = True
            if module.params['enable']:
                if module.params['enable'] != ds_now.enabled:
                    attr['enabled'] = module.params['enable']
                    mod_ds = True
            if password_required:
                if module.params['bind_password']:
                    attr['bind_password'] = module.params['bind_password']
                    mod_ds = True
                else:
                    module.fail_json(msg="'bind_password' must be provided for this task")
            if module.params['dstype'] == 'smb':
                if module.params['join_ou'] != ds_now.smb.join_ou:
                    attr['smb'] = {'join_ou': module.params['join_ou']}
                    mod_ds = True
        if mod_ds:
            changed = True
            if not module.check_mode:
                n_attr = DirectoryService(**attr)
                try:
                    blade.directory_services.update_directory_services(names=[module.params['dstype']], directory_service=n_attr)
                except Exception:
                    module.fail_json(msg='Failed to change {0} directory service.'.format(module.params['dstype']))
    except Exception:
        module.fail_json(msg='Failed to get current {0} directory service.'.format(module.params['dstype']))
    module.exit_json(changed=changed)