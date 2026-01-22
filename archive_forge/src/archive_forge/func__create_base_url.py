import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
def _create_base_url(self, auth, sess, endpoint):
    api_version = None
    if self.conf.barbican.barbican_api_version:
        api_version = self.conf.barbican.barbican_api_version
    elif getattr(auth, 'service_catalog', None):
        endpoint_data = auth.service_catalog.endpoint_data_for(service_type='key-manager', interface=self.conf.barbican.barbican_endpoint_type, region_name=self.conf.barbican.barbican_region_name)
        api_version = endpoint_data.api_version
    elif getattr(auth, 'get_discovery', None):
        discovery = auth.get_discovery(sess, url=endpoint)
        raw_data = discovery.raw_version_data()
        if len(raw_data) == 0:
            msg = _('Could not find discovery information for %s') % endpoint
            LOG.error(msg)
            raise exception.KeyManagerError(reason=msg)
        latest_version = raw_data[-1]
        api_version = latest_version.get('id')
    if endpoint[-1] != '/':
        endpoint += '/'
    base_url = urllib.parse.urljoin(endpoint, api_version)
    return base_url