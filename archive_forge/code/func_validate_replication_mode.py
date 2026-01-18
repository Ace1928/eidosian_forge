from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_replication_mode(self, replication):
    if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous':
        if replication['rpo'] is None:
            errormsg = "rpo is required together with 'asynchronous' replication_mode."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if replication['rpo'] < 5 or replication['rpo'] > 1440:
            errormsg = 'rpo value should be in range of 5 to 1440'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)