import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
def _get_conf_value_from_account_ref(self, netloc):
    try:
        ref_params = sutils.SwiftParams(self.conf, backend=self.backend_group).params
        self.user = ref_params[netloc]['user']
        self.key = ref_params[netloc]['key']
        netloc = ref_params[netloc]['auth_address']
        self.ssl_enabled = True
        if netloc != '':
            if netloc.startswith('http://'):
                self.ssl_enabled = False
                netloc = netloc[len('http://'):]
            elif netloc.startswith('https://'):
                netloc = netloc[len('https://'):]
    except KeyError:
        reason = _('Badly formed Swift URI. Credentials not found for account reference')
        LOG.info(reason)
        raise exceptions.BadStoreUri(message=reason)
    return netloc