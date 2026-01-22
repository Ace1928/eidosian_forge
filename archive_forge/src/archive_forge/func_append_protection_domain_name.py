from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def append_protection_domain_name(self, rcg_details):
    try:
        if 'protectionDomainId' in rcg_details and rcg_details['protectionDomainId']:
            pd_details = self.get_protection_domain(conn=self.powerflex_conn, protection_domain_id=rcg_details['protectionDomainId'])
            rcg_details['protectionDomainName'] = pd_details['name']
    except Exception as e:
        error_msg = "Updating replication consistency group details with protection domain name failed with error '%s'" % str(e)
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)