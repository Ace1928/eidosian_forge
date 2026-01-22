from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _remove_scsi_symlinks(self, devices_names):
    devices = ['/dev/' + dev for dev in devices_names]
    links = glob.glob('/dev/disk/by-id/scsi-*')
    unlink = []
    for link in links:
        try:
            if os.path.realpath(link) in devices:
                unlink.append(link)
        except OSError:
            continue
    if unlink:
        priv_rootwrap.unlink_root(*unlink, no_errors=True)