from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_storage_domain_and_all_template_disks(self, template):
    if self.param('template') is None:
        return None
    if self.param('storage_domain') is None:
        return None
    disks = list()
    for att in self._connection.follow_link(template.disk_attachments):
        disks.append(otypes.DiskAttachment(disk=otypes.Disk(id=att.disk.id, format=otypes.DiskFormat(self.param('disk_format')), sparse=self.param('disk_format') != 'raw', storage_domains=[otypes.StorageDomain(id=get_id_by_name(self._connection.system_service().storage_domains_service(), self.param('storage_domain')))])))
    return disks