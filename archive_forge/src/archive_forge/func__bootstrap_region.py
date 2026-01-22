import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_region(self):
    if self.region_id:
        try:
            PROVIDERS.catalog_api.create_region(region_ref={'id': self.region_id})
            LOG.info('Created region %s', self.region_id)
        except exception.Conflict:
            LOG.info('Region %s exists, skipping creation.', self.region_id)