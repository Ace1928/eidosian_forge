import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def add_pci_device(self, vm_name, vendor_id, product_id):
    """Adds the given PCI device to the given VM.

        :param vm_name: the name of the VM to which the PCI device will be
            attached to.
        :param vendor_id: the PCI device's vendor ID.
        :param product_id: the PCI device's product ID.
        :raises exceptions.PciDeviceNotFound: if there is no PCI device
            identifiable by the given vendor_id and product_id, or it was
            already assigned.
        """
    vmsettings = self._lookup_vm_check(vm_name)
    pci_setting_data = self._get_new_setting_data(self._PCI_EXPRESS_SETTING_DATA)
    pci_device = self._get_assignable_pci_device(vendor_id, product_id)
    pci_setting_data.HostResource = [pci_device.path_()]
    self._jobutils.add_virt_resource(pci_setting_data, vmsettings)