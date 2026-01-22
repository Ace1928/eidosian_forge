import calendar
import time
import eventlet
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_log import versionutils
from oslo_utils import encodeutils
from glance.common import crypt
from glance.common import exception
from glance.common import timeutils
from glance import context
import glance.db as db_api
from glance.i18n import _, _LC, _LE, _LI, _LW
def _delete_image_location_from_backend(self, image_id, loc_id, uri, backend=None):
    try:
        LOG.debug('Scrubbing image %s from a location.', image_id)
        try:
            if CONF.enabled_backends:
                self.store_api.delete(uri, backend, self.admin_context)
            else:
                self.store_api.delete_from_backend(uri, self.admin_context)
        except store_exceptions.NotFound:
            LOG.info(_LI("Image location for image '%s' not found in backend; Marking image location deleted in db."), image_id)
        if loc_id != '-':
            db_api.get_api().image_location_delete(self.admin_context, image_id, int(loc_id), 'deleted')
        LOG.info(_LI('Image %s is scrubbed from a location.'), image_id)
    except Exception as e:
        LOG.error(_LE('Unable to scrub image %(id)s from a location. Reason: %(exc)s '), {'id': image_id, 'exc': encodeutils.exception_to_unicode(e)})
        raise