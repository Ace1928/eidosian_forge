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
def _recover_node_startup_values(self, connection_properties, old_node_startups):
    node_startups = self._get_node_startup_values(connection_properties)
    for iqn, node_startup in node_startups.items():
        old_node_startup = old_node_startups.get(iqn, None)
        if old_node_startup and node_startup != old_node_startup:
            recover_connection = copy.deepcopy(connection_properties)
            recover_connection['target_iqn'] = iqn
            self._iscsiadm_update(recover_connection, 'node.startup', old_node_startup)