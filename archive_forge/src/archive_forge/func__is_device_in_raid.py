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
def _is_device_in_raid(self, device_path: str) -> bool:
    cmd = ['mdadm', '--examine', device_path]
    raid_expected = device_path + ':'
    try:
        lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
        for line in lines.split('\n'):
            if line == raid_expected:
                return True
            else:
                return False
    except putils.ProcessExecutionError:
        pass
    return False