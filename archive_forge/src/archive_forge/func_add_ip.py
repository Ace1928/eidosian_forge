from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def add_ip(self, sds_id, sds_ip_list):
    """Add IP to SDS
            :param sds_id: SDS ID
            :type sds_id: str
            :param sds_ip_list: List of one or more IP addresses and
                                their roles
            :type sds_ip_list: list[dict]
            :return: Boolean indicating if add IP operation is successful
        """
    try:
        if not self.module.check_mode:
            for ip in sds_ip_list:
                LOG.info('IP to add: %s', ip)
                self.powerflex_conn.sds.add_ip(sds_id=sds_id, sds_ip=ip)
                LOG.info('IP added successfully.')
        return True
    except Exception as e:
        error_msg = "Add IP to SDS '%s' operation failed with error '%s'" % (sds_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)