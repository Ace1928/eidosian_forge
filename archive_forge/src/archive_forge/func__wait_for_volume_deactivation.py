from __future__ import annotations
import math
import os
import re
from typing import Any, Callable, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
@utils.retry(retry_param=exception.VolumeNotDeactivated, retries=5, backoff_rate=2)
def _wait_for_volume_deactivation(self, name: str) -> None:
    LOG.debug('Checking to see if volume %s has been deactivated.', name)
    if self._lv_is_active(name):
        LOG.debug('Volume %s is still active.', name)
        raise exception.VolumeNotDeactivated(name=name)
    else:
        LOG.debug('Volume %s has been deactivated.', name)