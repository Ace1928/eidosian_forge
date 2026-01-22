from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
def _run_iscsi_session(self) -> tuple[str, str]:
    out, err = self._run_iscsiadm_bare(('-m', 'session'), check_exit_code=[0, 21, 255])
    LOG.debug('iscsi session list stdout=%(out)s stderr=%(err)s', {'out': out, 'err': err})
    return (out, err)