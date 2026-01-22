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
def _resize_on_write(self, image, image_size, bytes_written, chunk_length):
    """Handle the rbd resize when needed."""
    if image_size != 0 or self.size >= bytes_written + chunk_length:
        return self.size
    self.resize_amount = min(self.resize_amount * 2, 8 * units.Gi)
    new_size = self.size + self.resize_amount
    LOG.debug('resizing image to %s KiB' % (new_size / units.Ki))
    image.resize(new_size)
    return new_size