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
def _get_target_portals_from_iscsiadm_output(self, output: str) -> tuple[list[str], list[str]]:
    ips = []
    iqns = []
    for data in [line.split() for line in output.splitlines()]:
        if len(data) == 2 and data[1].startswith('iqn.'):
            ips.append(data[0].split(',')[0])
            iqns.append(data[1])
    return (ips, iqns)