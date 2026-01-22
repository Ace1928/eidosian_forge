import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _detach_disk_from_backing(self, session, backing, disk_device):
    LOG.debug('Reconfiguring backing VM: %(backing)s to remove disk: %(disk_device)s.', {'backing': backing, 'disk_device': disk_device})
    cf = session.vim.client.factory
    reconfig_spec = cf.create('ns0:VirtualMachineConfigSpec')
    spec = self._create_spec_for_disk_remove(session, disk_device)
    reconfig_spec.deviceChange = [spec]
    self._reconfigure_backing(session, backing, reconfig_spec)