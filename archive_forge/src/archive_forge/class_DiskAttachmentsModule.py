from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class DiskAttachmentsModule(DisksModule):

    def build_entity(self):
        return otypes.DiskAttachment(disk=super(DiskAttachmentsModule, self).build_entity(), interface=otypes.DiskInterface(self._module.params.get('interface')) if self._module.params.get('interface') else None, bootable=self._module.params.get('bootable'), active=self.param('activate'), read_only=self.param('read_only'), uses_scsi_reservation=self.param('uses_scsi_reservation'), pass_discard=self.param('pass_discard'))

    def update_check(self, entity):
        return super(DiskAttachmentsModule, self).update_check(follow_link(self._connection, entity.disk)) and equal(self._module.params.get('interface'), str(entity.interface)) and equal(self._module.params.get('bootable'), entity.bootable) and equal(self._module.params.get('pass_discard'), entity.pass_discard) and equal(self._module.params.get('read_only'), entity.read_only) and equal(self._module.params.get('uses_scsi_reservation'), entity.uses_scsi_reservation) and equal(self.param('activate'), entity.active)