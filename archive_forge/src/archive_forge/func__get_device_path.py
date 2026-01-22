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
def _get_device_path(self, connection_properties: dict) -> list:
    if self._get_transport() == 'default':
        return ['/dev/disk/by-path/ip-%s-iscsi-%s-lun-%s' % self._munge_portal(x) for x in self._get_all_targets(connection_properties)]
    else:
        device_list = []
        for x in self._get_all_targets(connection_properties):
            look_for_device = glob.glob('/dev/disk/by-path/*ip-%s-iscsi-%s-lun-%s' % self._munge_portal(x))
            if look_for_device:
                device_list.extend(look_for_device)
        return device_list