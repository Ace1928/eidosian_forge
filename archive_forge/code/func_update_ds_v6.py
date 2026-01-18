from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_ds_v6(module, array):
    """Update Directory Service"""
    changed = False
    ds_change = False
    password_required = False
    current_ds = []
    dirservlist = list(array.get_directory_services().items)
    for dirs in range(0, len(dirservlist)):
        if dirservlist[dirs].name == module.params['dstype']:
            current_ds = dirservlist[dirs]
    if module.params['uri'] and current_ds.uris is None:
        password_required = True
    if module.params['uri'] and current_ds.uris != module.params['uri']:
        uris = module.params['uri']
        ds_change = True
    else:
        uris = current_ds.uris
    base_dn = getattr(current_ds, 'base_dn', '')
    bind_user = getattr(current_ds, 'bind_user', '')
    cert = getattr(current_ds, 'ca_certificate', None)
    if module.params['base_dn'] and module.params['base_dn'] != base_dn:
        base_dn = module.params['base_dn']
        ds_change = True
    if module.params['enable'] != current_ds.enabled:
        ds_change = True
        if getattr(current_ds, 'bind_password', None) is None:
            password_required = True
    if module.params['bind_user'] is not None:
        if module.params['bind_user'] != bind_user:
            bind_user = module.params['bind_user']
            password_required = True
            ds_change = True
        elif module.params['force_bind_password']:
            password_required = True
            ds_change = True
    if module.params['bind_password'] is not None and password_required:
        bind_password = module.params['bind_password']
        ds_change = True
    if password_required and (not module.params['bind_password']):
        module.fail_json(msg="'bind_password' must be provided for this task")
    if module.params['dstype'] == 'management':
        if module.params['certificate'] is not None:
            if cert is None and module.params['certificate'] != '':
                cert = module.params['certificate']
                ds_change = True
            elif cert is None and module.params['certificate'] == '':
                pass
            elif module.params['certificate'] != cert:
                cert = module.params['certificate']
                ds_change = True
        if module.params['check_peer'] and (not cert):
            module.warn('Cannot check_peer without a CA certificate. Disabling check_peer')
            module.params['check_peer'] = False
        if module.params['check_peer'] != current_ds.check_peer:
            ds_change = True
        user_login = getattr(current_ds.management, 'user_login_attribute', '')
        user_object = getattr(current_ds.management, 'user_object_class', '')
        if module.params['user_object'] is not None and user_object != module.params['user_object']:
            user_object = module.params['user_object']
            ds_change = True
        if module.params['user_login'] is not None and user_login != module.params['user_login']:
            user_login = module.params['user_login']
            ds_change = True
        management = flasharray.DirectoryServiceManagement(user_login_attribute=user_login, user_object_class=user_object)
        if password_required:
            directory_service = flasharray.DirectoryService(uris=uris, base_dn=base_dn, bind_user=bind_user, bind_password=bind_password, enabled=module.params['enable'], services=module.params['dstype'], management=management, check_peer=module.params['check_peer'], ca_certificate=cert)
        else:
            directory_service = flasharray.DirectoryService(uris=uris, base_dn=base_dn, bind_user=bind_user, enabled=module.params['enable'], services=module.params['dstype'], management=management, check_peer=module.params['check_peer'], ca_certificate=cert)
    elif password_required:
        directory_service = flasharray.DirectoryService(uris=uris, base_dn=base_dn, bind_user=bind_user, bind_password=bind_password, enabled=module.params['enable'], services=module.params['dstype'])
    else:
        directory_service = flasharray.DirectoryService(uris=uris, base_dn=base_dn, bind_user=bind_user, enabled=module.params['enable'], services=module.params['dstype'])
    if ds_change:
        changed = True
        if not module.check_mode:
            res = array.patch_directory_services(names=[module.params['dstype']], directory_service=directory_service)
            if res.status_code != 200:
                module.fail_json(msg='{0} Directory Service failed. Error message: {1}'.format(module.params['dstype'].capitalize(), res.errors[0].message))
    module.exit_json(changed=changed)