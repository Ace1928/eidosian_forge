import contextlib
import logging
import math
import urllib
from eventlet import tpool
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import eventletutils
from oslo_utils import units
from glance_store import capabilities
from glance_store.common import utils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
from glance_store import location
def _unprotect_snapshot(self, image, snap_name):
    try:
        image.unprotect_snap(snap_name)
    except rbd.InvalidArgument:
        LOG.debug('Snapshot %s is unprotected already' % snap_name)