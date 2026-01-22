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
@utils.retry(exception.BrickException)
def _connect_to_iscsi_portal_unsafe(self, connection_properties: dict) -> tuple[Optional[str], Optional[bool]]:
    """Connect to an iSCSI portal-target an return the session id."""
    portal = connection_properties['target_portal'].split(',')[0]
    target_iqn = connection_properties['target_iqn']
    LOG.info('Trying to connect to iSCSI portal %s', portal)
    out, err = self._run_iscsiadm(connection_properties, (), check_exit_code=(0, 21, 255))
    if err:
        out_new, err_new = self._run_iscsiadm(connection_properties, ('--interface', self._get_transport(), '--op', 'new'), check_exit_code=(0, 6))
        if err_new:
            LOG.debug('Retrying to connect to iSCSI portal %s', portal)
            msg = _('Encountered database failure for %s.') % portal
            raise exception.BrickException(msg=msg)
    res = self._iscsiadm_update(connection_properties, 'node.session.scan', 'manual', check_exit_code=False)
    manual_scan = not res[1]
    initiator_utils.ISCSI_SUPPORTS_MANUAL_SCAN = manual_scan
    if connection_properties.get('auth_method'):
        self._iscsiadm_update(connection_properties, 'node.session.auth.authmethod', connection_properties['auth_method'])
        self._iscsiadm_update(connection_properties, 'node.session.auth.username', connection_properties['auth_username'])
        self._iscsiadm_update(connection_properties, 'node.session.auth.password', connection_properties['auth_password'])
    while True:
        sessions = self._get_iscsi_sessions_full()
        for s in sessions:
            if s[0] in self.VALID_SESSIONS_PREFIX and portal.lower() == s[2].lower() and (s[4] == target_iqn):
                return (str(s[1]), manual_scan)
        try:
            self._run_iscsiadm(connection_properties, ('--login',), check_exit_code=(0, 15, 255))
        except putils.ProcessExecutionError as p_err:
            LOG.warning('Failed to login iSCSI target %(iqn)s on portal %(portal)s (exit code %(err)s).', {'iqn': target_iqn, 'portal': portal, 'err': p_err.exit_code})
            return (None, None)
        self._iscsiadm_update(connection_properties, 'node.startup', 'automatic')