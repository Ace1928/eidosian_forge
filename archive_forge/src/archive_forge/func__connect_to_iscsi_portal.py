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
def _connect_to_iscsi_portal(self, connection_properties: dict) -> tuple[Optional[str], Optional[bool]]:
    """Safely connect to iSCSI portal-target and return the session id."""
    portal = connection_properties['target_portal'].split(',')[0]
    target_iqn = connection_properties['target_iqn']
    lock_name = f'connect_to_iscsi_portal-{portal}-{target_iqn}'
    method = base.synchronized(lock_name, external=True)(self._connect_to_iscsi_portal_unsafe)
    return method(connection_properties)