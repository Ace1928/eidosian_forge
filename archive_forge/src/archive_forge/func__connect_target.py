from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
@utils.retry(exception.VolumeDeviceNotFound, interval=2)
def _connect_target(self, target: Target) -> str:
    """Attach a specific target to present a volume on the system

        If we are already connected to any of the portals (and it's live) we
        send a rescan (because the backend may not support AER messages),
        otherwise we iterate through the portals trying to do an nvme-of
        connection.

        This method assumes that the controllers for the portals have already
        been set.  For example using the from_dictionary_parameter decorator
        in the NVMeOFConnProps class.

        Returns the path of the connected device.
        """
    connected = False
    missing_portals = []
    reconnecting_portals = []
    for portal in target.portals:
        state = portal.state
        if state == portal.LIVE:
            connected = True
            self.rescan(portal.controller)
        elif state == portal.MISSING:
            missing_portals.append(portal)
        elif state == portal.CONNECTING:
            LOG.debug('%s is reconnecting', portal)
            reconnecting_portals.append(portal)
        else:
            LOG.debug('%s exists but is %s', portal, state)
    do_multipath = self._do_multipath()
    if do_multipath or not connected:
        for portal in missing_portals:
            cmd = ['connect', '-a', portal.address, '-s', portal.port, '-t', portal.transport, '-n', target.nqn, '-Q', '128', '-l', '-1']
            if target.host_nqn:
                cmd.extend(['-q', target.host_nqn])
            try:
                self.run_nvme_cli(cmd)
                connected = True
            except putils.ProcessExecutionError as exc:
                if not (exc.exit_code in (70, errno.EALREADY) or (exc.exit_code == 1 and 'already connected' in exc.stderr + exc.stdout)):
                    LOG.error('Could not connect to %s: exit_code: %s, stdout: "%s", stderr: "%s",', portal, exc.exit_code, exc.stdout, exc.stderr)
                    continue
                LOG.warning('Race condition with some other application when connecting to %s, please check your system configuration.', portal)
                state = portal.state
                if state == portal.LIVE:
                    connected = True
                elif state == portal.CONNECTING:
                    reconnecting_portals.append(portal)
                else:
                    LOG.error('Ignoring %s due to unknown state (%s)', portal, state)
            if not do_multipath:
                break
    if not connected and reconnecting_portals:
        delay = self.TIME_TO_CONNECT + max((p.reconnect_delay for p in reconnecting_portals))
        LOG.debug('Waiting %s seconds for some nvme controllers to reconnect', delay)
        timeout = time.time() + delay
        while time.time() < timeout:
            time.sleep(1)
            if any((p.is_live for p in reconnecting_portals)):
                LOG.debug('Reconnected')
                connected = True
                break
        LOG.debug('No controller reconnected')
    if not connected:
        raise exception.VolumeDeviceNotFound(device=target.nqn)
    target.set_portals_controllers()
    dev_path = target.find_device()
    return dev_path