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
@utils.trace
def _connect_volume_replicated(self, connection_properties: NVMeOFConnProps) -> str:
    """Connect to a replicated volume and prepare the RAID

        Connection properties must contain all the necessary replica
        information, even if there is only 1 replica.

        Returns the /dev/md path

        Raises VolumeDeviceNotFound when cannot present the device in the
        system.
        """
    host_device_paths = []
    if not connection_properties.alias:
        raise exception.BrickException('Alias missing in connection info')
    for replica in connection_properties.targets:
        try:
            rep_host_device_path = self._connect_target(replica)
            host_device_paths.append(rep_host_device_path)
        except Exception as ex:
            LOG.error('_connect_target: %s', ex)
    if not host_device_paths:
        raise exception.VolumeDeviceNotFound(device=connection_properties.targets)
    if connection_properties.is_replicated:
        device_path = self._handle_replicated_volume(host_device_paths, connection_properties)
    else:
        device_path = self._handle_single_replica(host_device_paths, connection_properties.alias)
    if nvmeof_agent:
        nvmeof_agent.NVMeOFAgent.ensure_running(self)
    return device_path