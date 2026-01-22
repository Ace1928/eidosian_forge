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
def _create_container_if_missing(self, container, connection):
    """
        Creates a missing container in Swift if the
        ``swift_store_create_container_on_put`` option is set.

        :param container: Name of container to create
        :param connection: Connection to swift service
        """
    if self.backend_group:
        store_conf = getattr(self.conf, self.backend_group)
    else:
        store_conf = self.conf.glance_store
    try:
        connection.head_container(container)
    except swiftclient.ClientException as e:
        if e.http_status == http.client.NOT_FOUND:
            if store_conf.swift_store_create_container_on_put:
                try:
                    msg = _LI('Creating swift container %(container)s') % {'container': container}
                    LOG.info(msg)
                    connection.put_container(container)
                except swiftclient.ClientException as e:
                    msg = _('Failed to add container to Swift.\nGot error from Swift: %s.') % encodeutils.exception_to_unicode(e)
                    raise glance_store.BackendException(msg)
            else:
                msg = _('The container %(container)s does not exist in Swift. Please set the swift_store_create_container_on_put option to add container to Swift automatically.') % {'container': container}
                raise glance_store.BackendException(msg)
        else:
            raise