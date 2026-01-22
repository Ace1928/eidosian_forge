from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
@utils.retry(exception.VolumeDeviceNotFound)
def _connect_single_volume(self, connection_properties: dict) -> Optional[dict[str, str]]:
    """Connect to a volume using a single path."""
    data: dict[str, Any] = {'stop_connecting': False, 'num_logins': 0, 'failed_logins': 0, 'stopped_threads': 0, 'found_devices': [], 'just_added_devices': []}
    for props in self._iterate_all_targets(connection_properties):
        self._connect_vol(self.device_scan_attempts, props, data)
        found_devs = data['found_devices']
        if found_devs:
            for __ in range(10):
                wwn = self._linuxscsi.get_sysfs_wwn(found_devs)
                if wwn:
                    break
                time.sleep(1)
            else:
                LOG.debug('Could not find the WWN for %s.', found_devs[0])
            return self._get_connect_result(connection_properties, wwn, found_devs)
        ips_iqns_luns = [(props['target_portal'], props['target_iqn'], props['target_lun'])]
        self._cleanup_connection(props, ips_iqns_luns, force=True, ignore_errors=True)
        data.update(num_logins=0, failed_logins=0, found_devices=[])
    raise exception.VolumeDeviceNotFound(device='')