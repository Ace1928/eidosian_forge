import contextlib
import errno
import importlib
import logging
import math
import os
import shlex
import socket
import time
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from keystoneauth1 import token_endpoint as ksa_token_endpoint
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_utils import strutils
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import capabilities
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
import glance_store.location
from the service catalog, and current context's user and project are used.
def _wait_volume_status(self, volume, status_transition, status_expected):
    max_recheck_wait = 15
    timeout = self.store_conf.cinder_state_transition_timeout
    volume = volume.manager.get(volume.id)
    tries = 0
    elapsed = 0
    while volume.status == status_transition:
        if elapsed >= timeout:
            msg = _('Timeout while waiting while volume %(volume_id)s status is %(status)s.') % {'volume_id': volume.id, 'status': status_transition}
            LOG.error(msg)
            raise exceptions.BackendException(msg)
        wait = min(0.5 * 2 ** tries, max_recheck_wait)
        time.sleep(wait)
        tries += 1
        elapsed += wait
        volume = volume.manager.get(volume.id)
    if volume.status != status_expected:
        msg = _('The status of volume %(volume_id)s is unexpected: status = %(status)s, expected = %(expected)s.') % {'volume_id': volume.id, 'status': volume.status, 'expected': status_expected}
        LOG.error(msg)
        raise exceptions.BackendException(msg)
    return volume