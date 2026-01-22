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
def _get_pci_num(self, hba: Optional[dict]) -> tuple:
    platform = None
    if hba is not None:
        if 'device_path' in hba:
            device_path = hba['device_path'].split('/')
            has_platform = len(device_path) > 3 and device_path[3] == 'platform'
            for index, value in enumerate(device_path):
                if has_platform and value.startswith('pci'):
                    platform = 'platform-%s' % device_path[index - 1]
                if value.startswith('net') or value.startswith('host'):
                    return (platform, device_path[index - 1])
    return (None, None)