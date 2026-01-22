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
def _connect_vol(self, rescans: int, props: dict, data: dict[str, Any]) -> None:
    """Make a connection to a volume, send scans and wait for the device.

        This method is specifically designed to support multithreading and
        share the results via a shared dictionary with fixed keys, which is
        thread safe.

        Since the heaviest operations are run via subprocesses we don't worry
        too much about the GIL or how the eventlets will handle the context
        switching.

        The method will only try to log in once, since iscsid's initiator
        already tries 8 times by default to do the login, or whatever value we
        have as node.session.initial_login_retry_max in our system.

        Shared dictionary has the following keys:
        - stop_connecting: When the caller wants us to stop the rescans
        - num_logins: Count of how many threads have successfully logged in
        - failed_logins: Count of how many threads have failed to log in
        - stopped_threads: How many threads have finished.  This may be
                           different than num_logins + failed_logins, since
                           some threads may still be waiting for a device.
        - found_devices: List of devices the connections have found
        - just_added_devices: Devices that have been found and still have not
                              been processed by the main thread that manages
                              all the connecting threads.

        :param rescans: Number of rescans to perform before giving up.
        :param props: Properties of the connection.
        :param data: Shared data.
        """
    device = hctl = None
    portal = props['target_portal']
    try:
        session, manual_scan = self._connect_to_iscsi_portal(props)
    except Exception:
        LOG.exception('Exception connecting to %s', portal)
        session = None
    if session:
        do_scans = rescans > 0 or manual_scan
        if manual_scan:
            num_rescans = -1
            seconds_next_scan = 0
        else:
            num_rescans = 0
            seconds_next_scan = 4
        data['num_logins'] += 1
        LOG.debug('Connected to %s', portal)
        while do_scans:
            try:
                if not hctl:
                    hctl = self._linuxscsi.get_hctl(session, props['target_lun'])
                if hctl:
                    if seconds_next_scan <= 0:
                        num_rescans += 1
                        self._linuxscsi.scan_iscsi(*hctl)
                        seconds_next_scan = (num_rescans + 2) ** 2
                    device = self._linuxscsi.device_name_by_hctl(session, hctl)
                    if device:
                        break
            except Exception:
                LOG.exception('Exception scanning %s', portal)
                pass
            do_scans = num_rescans <= rescans and (not (device or data['stop_connecting']))
            if do_scans:
                time.sleep(1)
                seconds_next_scan -= 1
        if device:
            LOG.debug('Connected to %s using %s', device, strutils.mask_password(props))
        else:
            LOG.warning('LUN %(lun)s on iSCSI portal %(portal)s not found on sysfs after logging in.', {'lun': props['target_lun'], 'portal': portal})
    else:
        LOG.warning('Failed to connect to iSCSI portal %s.', portal)
        data['failed_logins'] += 1
    if device:
        data['found_devices'].append(device)
        data['just_added_devices'].append(device)
    data['stopped_threads'] += 1