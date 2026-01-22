import os
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import failure
from glance.common import exception
from glance.i18n import _, _LE
def _copy_to_staging_store(self, loc):
    store_backend = loc['metadata'].get('store')
    image_data, size = store_api.get(loc['url'], store_backend)
    msg = 'Found image, copying it in staging area'
    LOG.debug(msg)
    return self.staging_store.add(self.image_id, image_data, size)[0]