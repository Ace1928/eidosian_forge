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
def _discover_iscsi_portals(self, connection_properties: dict) -> list:
    out = None
    iscsi_transport = 'iser' if self._get_transport() == 'iser' else 'default'
    if connection_properties.get('discovery_auth_method'):
        try:
            self._run_iscsiadm_update_discoverydb(connection_properties, iscsi_transport)
        except putils.ProcessExecutionError as exception:
            if exception.exit_code == 6:
                self._run_iscsiadm_bare(['-m', 'discoverydb', '-t', 'sendtargets', '-p', connection_properties['target_portal'], '-I', iscsi_transport, '--op', 'new'], check_exit_code=[0, 255])
                self._run_iscsiadm_update_discoverydb(connection_properties)
            else:
                LOG.error('Unable to find target portal: %(target_portal)s.', {'target_portal': connection_properties['target_portal']})
                raise
        old_node_startups = self._get_node_startup_values(connection_properties)
        out = self._run_iscsiadm_bare(['-m', 'discoverydb', '-t', 'sendtargets', '-I', iscsi_transport, '-p', connection_properties['target_portal'], '--discover'], check_exit_code=[0, 255])[0] or ''
        self._recover_node_startup_values(connection_properties, old_node_startups)
    else:
        old_node_startups = self._get_node_startup_values(connection_properties)
        out = self._run_iscsiadm_bare(['-m', 'discovery', '-t', 'sendtargets', '-I', iscsi_transport, '-p', connection_properties['target_portal']], check_exit_code=[0, 255])[0] or ''
        self._recover_node_startup_values(connection_properties, old_node_startups)
    ips, iqns = self._get_target_portals_from_iscsiadm_output(out)
    luns = self._get_luns(connection_properties, iqns)
    return list(zip(ips, iqns, luns))