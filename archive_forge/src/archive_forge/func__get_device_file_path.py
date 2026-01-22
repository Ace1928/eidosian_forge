from oslo_log import log as logging
from os_brick import initiator
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
def _get_device_file_path(self, pci_num, target_wwn, lun):
    host_device = ['/dev/disk/by-path/ccw-%s-zfcp-%s:%s' % (pci_num, target_wwn, self._get_lun_string(lun)), '/dev/disk/by-path/ccw-%s-fc-%s-lun-%s' % (pci_num, target_wwn, lun), '/dev/disk/by-path/ccw-%s-fc-%s-lun-%s' % (pci_num, target_wwn, self._get_lun_string(lun))]
    return host_device