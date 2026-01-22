import logging
import urllib
from oslo_config import cfg
from oslo_utils import encodeutils
import requests
from glance_store import capabilities
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LI
import glance_store.location
@staticmethod
def _check_store_uri(conn, loc):
    if conn.status_code >= 400:
        if conn.status_code == requests.codes.not_found:
            reason = _('HTTP datastore could not find image at URI.')
            LOG.debug(reason)
            raise exceptions.NotFound(message=reason)
        reason = _('HTTP URL %(url)s returned a %(status)s status code. \nThe response body:\n%(body)s') % {'url': loc.path, 'status': conn.status_code, 'body': conn.text}
        LOG.debug(reason)
        raise exceptions.BadStoreUri(message=reason)
    if conn.is_redirect and conn.status_code not in (301, 302):
        reason = (_('The HTTP URL %(url)s attempted to redirect with an invalid %(status)s status code.'), {'url': loc.path, 'status': conn.status_code})
        LOG.info(reason)
        raise exceptions.BadStoreUri(message=reason)