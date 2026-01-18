from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_s3user(module, blade):
    """Update Object Store User"""
    changed = False
    exists = False
    s3user_facts = {}
    user = module.params['account'] + '/' + module.params['name']
    if module.params['access_key'] or module.params['imported_key']:
        key_count = 0
        keys = blade.object_store_access_keys.list_object_store_access_keys()
        for key in range(0, len(keys.items)):
            if module.params['imported_key']:
                versions = blade.api_version.list_versions().versions
                if IMPORT_KEY_API_VERSION in versions:
                    if keys.items[key].name == module.params['imported_key']:
                        module.warn('Imported key provided already belongs to a user')
                        exists = True
            if keys.items[key].user.name == user:
                key_count += 1
        if not exists:
            if key_count < 2:
                try:
                    if module.params['access_key'] and module.params['imported_key']:
                        module.warn("'access_key: true' overrides imported keys")
                    if module.params['access_key']:
                        if key_count == 0 or (key_count >= 1 and module.params['multiple_keys']):
                            changed = True
                            if not module.check_mode:
                                result = blade.object_store_access_keys.create_object_store_access_keys(object_store_access_key=ObjectStoreAccessKey(user={'name': user}))
                                s3user_facts['fb_s3user'] = {'user': user, 'access_key': result.items[0].secret_access_key, 'access_id': result.items[0].name}
                    elif IMPORT_KEY_API_VERSION in versions:
                        changed = True
                        if not module.check_mode:
                            blade.object_store_access_keys.create_object_store_access_keys(names=[module.params['imported_key']], object_store_access_key=ObjectStoreAccessKeyPost(user={'name': user}, secret_access_key=module.params['imported_secret']))
                except Exception:
                    if module.params['imported_key']:
                        module.fail_json(msg='Object Store User {0}: Access Key import failed'.format(user))
                    else:
                        module.fail_json(msg='Object Store User {0}: Access Key creation failed'.format(user))
            else:
                module.warn('Object Store User {0}: Maximum Access Key count reached'.format(user))
    module.exit_json(changed=changed, s3user_info=s3user_facts)