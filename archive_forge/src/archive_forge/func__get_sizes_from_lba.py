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
@staticmethod
def _get_sizes_from_lba(ns_data: dict) -> tuple[Optional[int], Optional[int]]:
    """Return size in bytes and the nsze of the volume from NVMe NS data.

        nsze is the namespace size that defines the total size of the namespace
        in logical blocks (LBA 0 through n-1), as per NVMe-oF specs.

        Returns a tuple of nsze and size
        """
    try:
        lbads = ns_data['lbafs'][0]['ds']
        if len(ns_data['lbafs']) != 1 or lbads < 9:
            LOG.warning('Cannot calculate new size with LBAs')
            return (None, None)
        nsze = ns_data['nsze']
        new_size = nsze * (1 << lbads)
    except Exception:
        return (None, None)
    LOG.debug('New volume size is %s and nsze is %s', new_size, nsze)
    return (nsze, new_size)