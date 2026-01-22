from __future__ import annotations
import functools
import inspect
import logging as py_logging
import os
import time
from typing import Any, Callable, Optional, Type, Union   # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import executor
from os_brick.i18n import _
from os_brick.privileged import nvmeof as priv_nvme
from os_brick.privileged import rootwrap as priv_rootwrap
import tenacity  # noqa
def _symlink_name_from_device_path(device_path):
    """Generate symlink absolute path for encrypted devices.

    The symlink's basename will contain the original device name so we can
    reconstruct it afterwards on disconnect.

    Being able to restore the original device name may be important for some
    connectors, because the system may have multiple devices for the same
    connection information (for example if a controller came back to life after
    having network issues and an auto scan presented the device) and if we
    reuse an existing symlink created by udev we wouldn't know which one was
    actually used.

    The symlink will be created under the /dev/disk/by-id directory and will
    prefix the name with os-brick- and then continue with the full device path
    that was passed (replacing '/' with '+')
    """
    encoded_device = device_path.replace('/', '+')
    return CUSTOM_LINK_PREFIX + encoded_device