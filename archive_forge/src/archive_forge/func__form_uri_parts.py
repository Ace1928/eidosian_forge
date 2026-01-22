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
def _form_uri_parts(self, netloc, path):
    if netloc != '':
        if '@' in netloc:
            creds, netloc = netloc.split('@')
        else:
            creds = None
    else:
        if '@' in path:
            creds, path = path.split('@')
        else:
            creds = None
        netloc = path[0:path.find('/')].strip('/')
        path = path[path.find('/'):].strip('/')
    if creds:
        cred_parts = creds.split(':')
        if len(cred_parts) < 2:
            reason = _('Badly formed credentials in Swift URI.')
            LOG.info(reason)
            raise exceptions.BadStoreUri(message=reason)
        key = cred_parts.pop()
        user = ':'.join(cred_parts)
        creds = urllib.parse.unquote(creds)
        try:
            self.user, self.key = creds.rsplit(':', 1)
        except exceptions.BadStoreConfiguration:
            self.user = urllib.parse.unquote(user)
            self.key = urllib.parse.unquote(key)
    else:
        self.user = None
        self.key = None
    return (netloc, path)