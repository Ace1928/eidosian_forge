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
@utils.retry(exception.VolumeDeviceNotFound)
def wait_for_path(self, volume_path):
    """Wait for a path to show up."""
    LOG.debug('Checking to see if %s exists yet.', volume_path)
    if not os.path.exists(volume_path):
        LOG.debug("%(path)s doesn't exists yet.", {'path': volume_path})
        raise exception.VolumeDeviceNotFound(device=volume_path)
    else:
        LOG.debug('%s has shown up.', volume_path)