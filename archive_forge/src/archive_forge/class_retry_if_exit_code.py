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
class retry_if_exit_code(tenacity.retry_if_exception):
    """Retry on ProcessExecutionError specific exit codes."""

    def __init__(self, codes: Union[int, tuple[int, ...]]):
        self.codes = (codes,) if isinstance(codes, int) else codes
        super(retry_if_exit_code, self).__init__(self._check_exit_code)

    def _check_exit_code(self, exc: Type[Exception]) -> bool:
        return bool(exc) and isinstance(exc, processutils.ProcessExecutionError) and (exc.exit_code in self.codes)