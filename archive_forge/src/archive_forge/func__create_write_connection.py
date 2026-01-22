import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def _create_write_connection(self, method, url, file_size=None, cookies=None, overwrite=None, content_type=None, cacerts=False, ssl_thumbprint=None):
    """Create HTTP connection to write to VMDK file."""
    LOG.debug('Creating HTTP connection to write to file with size = %(file_size)d and URL = %(url)s.', {'file_size': file_size, 'url': url})
    try:
        conn = self._create_connection(url, method, cacerts, ssl_thumbprint, cookies=cookies)
        if file_size:
            conn.putheader('Content-Length', str(file_size))
        if overwrite:
            conn.putheader('Overwrite', overwrite)
        if content_type:
            conn.putheader('Content-Type', content_type)
        conn.endheaders()
        return conn
    except requests.RequestException as excep:
        excep_msg = _('Error occurred while creating HTTP connection to write to VMDK file with URL = %s.') % url
        LOG.exception(excep_msg)
        raise exceptions.VimConnectionException(excep_msg, excep)