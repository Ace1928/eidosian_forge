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
def _vg_exists(self) -> bool:
    """Simple check to see if VG exists.

        :returns: True if vg specified in object exists, else False

        """
    exists = False
    cmd = LVM.LVM_CMD_PREFIX + ['vgs', '--noheadings', '-o', 'name', self.vg_name]
    out, _err = self._execute(*cmd, root_helper=self._root_helper, run_as_root=True)
    if out is not None:
        volume_groups = out.split()
        if self.vg_name in volume_groups:
            exists = True
    return exists