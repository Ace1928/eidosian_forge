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
def _get_fs_type(self, device_path: str) -> Optional[str]:
    cmd = ['blkid', device_path, '-s', 'TYPE', '-o', 'value']
    LOG.debug('[!] cmd = %s', cmd)
    fs_type = None
    lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper, check_exit_code=False)
    fs_type = lines.split('\n')[0]
    return fs_type or None