import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _get_disk_device(self, session, backing):
    hardware_devices = session.invoke_api(vim_util, 'get_object_property', session.vim, backing, 'config.hardware.device')
    if hardware_devices.__class__.__name__ == 'ArrayOfVirtualDevice':
        hardware_devices = hardware_devices.VirtualDevice
    for device in hardware_devices:
        if device.__class__.__name__ == 'VirtualDisk':
            return device