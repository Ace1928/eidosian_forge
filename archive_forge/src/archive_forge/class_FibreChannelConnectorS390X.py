from oslo_log import log as logging
from os_brick import initiator
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
class FibreChannelConnectorS390X(fibre_channel.FibreChannelConnector):
    """Connector class to attach/detach Fibre Channel volumes on S390X arch."""
    platform = initiator.PLATFORM_S390

    def __init__(self, root_helper, driver=None, execute=None, use_multipath=False, device_scan_attempts=initiator.DEVICE_SCAN_ATTEMPTS_DEFAULT, *args, **kwargs):
        super(FibreChannelConnectorS390X, self).__init__(root_helper, *args, driver=driver, execute=execute, device_scan_attempts=device_scan_attempts, **kwargs)
        LOG.debug('Initializing Fibre Channel connector for S390')
        self._linuxfc = linuxfc.LinuxFibreChannelS390X(root_helper, execute)
        self.use_multipath = use_multipath

    def set_execute(self, execute):
        super(FibreChannelConnectorS390X, self).set_execute(execute)
        self._linuxscsi.set_execute(execute)
        self._linuxfc.set_execute(execute)

    def _get_host_devices(self, possible_devs):
        host_devices = []
        for pci_num, target_wwn, lun in possible_devs:
            host_device = self._get_device_file_path(pci_num, target_wwn, lun)
            self._linuxfc.configure_scsi_device(pci_num, target_wwn, self._get_lun_string(lun))
            host_devices.extend(host_device)
        return host_devices

    def _get_lun_string(self, lun):
        target_lun = 0
        if lun <= 65535:
            target_lun = '0x%04x000000000000' % lun
        elif lun <= 4294967295:
            target_lun = '0x%08x00000000' % lun
        return target_lun

    def _get_device_file_path(self, pci_num, target_wwn, lun):
        host_device = ['/dev/disk/by-path/ccw-%s-zfcp-%s:%s' % (pci_num, target_wwn, self._get_lun_string(lun)), '/dev/disk/by-path/ccw-%s-fc-%s-lun-%s' % (pci_num, target_wwn, lun), '/dev/disk/by-path/ccw-%s-fc-%s-lun-%s' % (pci_num, target_wwn, self._get_lun_string(lun))]
        return host_device

    def _remove_devices(self, connection_properties, devices, device_info, force, exc):
        hbas = self._linuxfc.get_fc_hbas_info()
        targets = connection_properties['targets']
        addressing_mode = connection_properties.get('addressing_mode')
        possible_devs = self._get_possible_devices(hbas, targets, addressing_mode)
        for platform, pci_num, target_wwn, lun in possible_devs:
            target_lun = self._get_lun_string(lun)
            with exc.context(force, 'Removing device %s:%s:%s failed', pci_num, target_wwn, target_lun):
                self._linuxfc.deconfigure_scsi_device(pci_num, target_wwn, target_lun)