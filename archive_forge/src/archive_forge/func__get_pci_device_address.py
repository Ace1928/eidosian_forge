import re
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import hostutils
from oslo_log import log as logging
def _get_pci_device_address(self, pci_device_path):
    pnp_device = self._conn_cimv2.Win32_PnPEntity(DeviceID=pci_device_path)
    return_code, pnp_device_props = pnp_device[0].GetDeviceProperties()
    if return_code:
        LOG.debug('Failed to get PnP Device Properties for the PCI device: %(pci_dev)s. (return_code=%(return_code)s', {'pci_dev': pci_device_path, 'return_code': return_code})
        return None
    pnp_props = {prop.KeyName: prop.Data for prop in pnp_device_props}
    location_info = pnp_props.get('DEVPKEY_Device_LocationInfo')
    slot = pnp_props.get('DEVPKEY_Device_Address')
    try:
        [bus, domain, funct] = self._PCI_ADDRESS_REGEX.findall(location_info)
        address = '%04x:%02x:%02x.%1x' % (int(domain), int(bus), int(slot), int(funct))
        return address
    except Exception as ex:
        LOG.debug('Failed to get PCI device address. Device path: %(device_path)s. Exception: %(ex)s', {'device_path': pci_device_path, 'ex': ex})
        return None