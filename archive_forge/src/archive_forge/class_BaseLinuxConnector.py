from __future__ import annotations
import functools
import glob
import os
import typing
from typing import Optional
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import reflection
from oslo_utils import timeutils
from os_brick import exception
from os_brick import initiator
from os_brick.initiator import host_driver
from os_brick.initiator import initiator_connector
from os_brick.initiator import linuxscsi
from os_brick import utils
class BaseLinuxConnector(initiator_connector.InitiatorConnector):
    os_type = initiator.OS_TYPE_LINUX

    def __init__(self, root_helper: str, driver=None, execute=None, *args, **kwargs):
        self._linuxscsi = linuxscsi.LinuxSCSI(root_helper, execute=execute)
        if not driver:
            driver = host_driver.HostDriver()
        self.set_driver(driver)
        super(BaseLinuxConnector, self).__init__(root_helper, *args, execute=execute, **kwargs)

    @staticmethod
    def get_connector_properties(root_helper: str, *args, **kwargs) -> dict:
        """The generic connector properties."""
        multipath = kwargs['multipath']
        enforce_multipath = kwargs['enforce_multipath']
        props = {}
        props['multipath'] = multipath and linuxscsi.LinuxSCSI.is_multipath_running(enforce_multipath, root_helper, execute=kwargs.get('execute'))
        return props

    def check_valid_device(self, path: str, run_as_root: bool=True) -> bool:
        return utils.check_valid_device(self, path)

    def get_all_available_volumes(self, connection_properties: Optional[dict]=None) -> list:
        volumes = []
        path = self.get_search_path()
        if path:
            if os.path.isdir(path):
                path_items = [path, '/*']
                file_filter = ''.join(path_items)
                volumes = glob.glob(file_filter)
        return volumes

    def _discover_mpath_device(self, device_wwn: str, connection_properties: dict, device_name: str) -> tuple[str, str]:
        """This method discovers a multipath device.

        Discover a multipath device based on a defined connection_property
        and a device_wwn and return the multipath_id and path of the multipath
        enabled device if there is one.
        """
        path = self._linuxscsi.find_multipath_device_path(device_wwn)
        device_path = None
        multipath_id = None
        if path is None:
            device_realpath = os.path.realpath(device_name)
            mpath_info = self._linuxscsi.find_multipath_device(device_realpath)
            if mpath_info:
                device_path = mpath_info['device']
                multipath_id = device_wwn
            else:
                device_path = device_name
                LOG.debug('Unable to find multipath device name for volume. Using path %(device)s for volume.', {'device': device_path})
        else:
            device_path = path
            multipath_id = device_wwn
        if connection_properties.get('access_mode', '') != 'ro':
            try:
                self._linuxscsi.wait_for_rw(device_wwn, device_path)
            except exception.BlockDeviceReadOnly:
                LOG.warning('Block device %s is still read-only. Continuing anyway.', device_path)
        device_path = typing.cast(str, device_path)
        multipath_id = typing.cast(str, multipath_id)
        return (device_path, multipath_id)