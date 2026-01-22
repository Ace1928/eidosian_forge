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
def _format_lun_id(self, lun_id):
    lun_id = int(lun_id)
    if lun_id < 256:
        return lun_id
    else:
        return '0x%04x%04x00000000' % (lun_id & 65535, lun_id >> 16 & 65535)