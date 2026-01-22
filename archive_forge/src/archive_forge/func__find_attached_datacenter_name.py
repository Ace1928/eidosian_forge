from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _find_attached_datacenter_name(self, sd_name):
    """
        Finds the name of the datacenter that a given
        storage domain is attached to.

        Args:
            sd_name (str): Storage Domain name

        Returns:
            str: Data Center name

        Raises:
            Exception: In case storage domain in not attached to
                an active Datacenter
        """
    dcs_service = self._connection.system_service().data_centers_service()
    dc = search_by_attributes(dcs_service, storage=sd_name)
    if dc is None:
        raise Exception("Can't bring storage to state `%s`, because it seems thatit is not attached to any datacenter" % self.param('state'))
    elif dc.status == dcstatus.UP:
        return dc.name
    else:
        raise Exception("Can't bring storage to state `%s`, because Datacenter %s is not UP" % (self.param('state'), dc.name))