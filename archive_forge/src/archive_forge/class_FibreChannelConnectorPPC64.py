from oslo_log import log as logging
from os_brick import initiator
from os_brick.initiator.connectors import fibre_channel
class FibreChannelConnectorPPC64(fibre_channel.FibreChannelConnector):
    """Connector class to attach/detach Fibre Channel volumes on PPC64 arch."""
    platform = initiator.PLATFORM_PPC64

    def __init__(self, root_helper, driver=None, execute=None, use_multipath=False, device_scan_attempts=initiator.DEVICE_SCAN_ATTEMPTS_DEFAULT, *args, **kwargs):
        super(FibreChannelConnectorPPC64, self).__init__(root_helper, *args, driver=driver, execute=execute, device_scan_attempts=device_scan_attempts, **kwargs)
        self.use_multipath = use_multipath

    def set_execute(self, execute):
        super(FibreChannelConnectorPPC64, self).set_execute(execute)
        self._linuxscsi.set_execute(execute)
        self._linuxfc.set_execute(execute)

    def _get_host_devices(self, possible_devs, lun):
        host_devices = set()
        for pci_num, target_wwn in possible_devs:
            host_device = '/dev/disk/by-path/fc-%s-lun-%s' % (target_wwn, self._linuxscsi.process_lun_id(lun))
            host_devices.add(host_device)
        return list(host_devices)