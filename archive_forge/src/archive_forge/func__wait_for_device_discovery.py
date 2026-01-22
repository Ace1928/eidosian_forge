from __future__ import annotations
import os
import typing
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator import linuxfc
from os_brick import utils
def _wait_for_device_discovery(host_devices: list[str]) -> None:
    for device in host_devices:
        LOG.debug('Looking for Fibre Channel dev %(device)s', {'device': device})
        if os.path.exists(device) and self.check_valid_device(device):
            self.host_device = device
            self.device_name = os.path.realpath(device)
            raise loopingcall.LoopingCallDone()
    if self.tries >= self.device_scan_attempts:
        LOG.error('Fibre Channel volume device not found.')
        raise exception.NoFibreChannelVolumeDeviceFound()
    LOG.info('Fibre Channel volume device not yet found. Will rescan & retry.  Try number: %(tries)s.', {'tries': self.tries})
    self._linuxfc.rescan_hosts(hbas, connection_properties)
    self.tries = self.tries + 1