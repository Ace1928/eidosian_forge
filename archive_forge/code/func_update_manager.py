from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_manager(module, blade):
    """Update SNMP Manager"""
    changed = False
    try:
        mgr = blade.snmp_managers.list_snmp_managers(names=[module.params['name']])
    except Exception:
        module.fail_json(msg='Failed to get configuration for SNMP manager {0}.'.format(module.params['name']))
    current_attr = {'community': mgr.items[0].v2c.community, 'notification': mgr.items[0].notification, 'host': mgr.items[0].host, 'version': mgr.items[0].version, 'auth_passphrase': mgr.items[0].v3.auth_passphrase, 'auth_protocol': mgr.items[0].v3.auth_protocol, 'privacy_passphrase': mgr.items[0].v3.privacy_passphrase, 'privacy_protocol': mgr.items[0].v3.privacy_protocol, 'user': mgr.items[0].v3.user}
    new_attr = {'community': module.params['community'], 'notification': module.params['notification'], 'host': module.params['host'], 'version': module.params['version'], 'auth_passphrase': module.params['auth_passphrase'], 'auth_protocol': module.params['auth_protocol'], 'privacy_passphrase': module.params['privacy_passphrase'], 'privacy_protocol': module.params['privacy_protocol'], 'user': module.params['user']}
    if current_attr != new_attr:
        changed = True
        if not module.check_mode:
            if new_attr['version'] == 'v2c':
                updated_v2c_attrs = SnmpV2c(community=new_attr['community'])
                updated_v2c_manager = SnmpManager(host=new_attr['host'], notification=new_attr['notification'], version='v2c', v2c=updated_v2c_attrs)
                try:
                    blade.snmp_managers.update_snmp_managers(names=[module.params['name']], snmp_manager=updated_v2c_manager)
                except Exception:
                    module.fail_json(msg='Failed to update v2c SNMP manager {0}.'.format(module.params['name']))
            else:
                updated_v3_attrs = SnmpV3(auth_protocol=new_attr['auth_protocol'], auth_passphrase=new_attr['auth_passphrase'], privacy_protocol=new_attr['privacy_protocol'], privacy_passphrase=new_attr['privacy_passphrase'], user=new_attr['user'])
                updated_v3_manager = SnmpManager(host=new_attr['host'], notification=new_attr['notification'], version='v3', v3=updated_v3_attrs)
                try:
                    blade.snmp_managers.update_snmp_managers(names=[module.params['name']], snmp_manager=updated_v3_manager)
                except Exception:
                    module.fail_json(msg='Failed to update v3 SNMP manager {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)