from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_cluster(self):
    if self.param('cluster') is not None:
        return self.param('cluster')
    elif self.param('snapshot_name') is not None and self.param('snapshot_vm') is not None:
        vms_service = self._connection.system_service().vms_service()
        vm = search_by_name(vms_service, self.param('snapshot_vm'))
        return self._connection.system_service().clusters_service().cluster_service(vm.cluster.id).get().name