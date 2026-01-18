from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_rpo(self, replication):
    if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous' and (replication['rpo'] is None):
        errormsg = "rpo is required together with 'asynchronous' replication_mode."
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if (replication['rpo'] and (replication['rpo'] < 5 or replication['rpo'] > 1440)) and (replication['replication_mode'] and replication['replication_mode'] != 'manual' or (not replication['replication_mode'] and replication['rpo'] != -1)):
        errormsg = 'rpo value should be in range of 5 to 1440'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)