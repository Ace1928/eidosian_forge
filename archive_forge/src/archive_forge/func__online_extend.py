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
def _online_extend(self, client, volume, write_props):
    with self._open_cinder_volume(client, volume, 'wb') as f:
        conn = self.volume_connector_map[volume.id]
        while write_props.need_extend:
            self._write_data(f, write_props)
            if write_props.need_extend:
                write_props.size_gb = self._call_online_extend(client, volume, write_props.size_gb)
                conn.extend_volume()