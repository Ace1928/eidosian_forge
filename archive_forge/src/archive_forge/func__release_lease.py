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
def _release_lease(self):
    """Release the lease

        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
    LOG.debug('Getting lease state for %s.', self._url)
    state = self._session.invoke_api(vim_util, 'get_object_property', self._session.vim, self._lease, 'state')
    LOG.debug('Lease for %(url)s is in state: %(state)s.', {'url': self._url, 'state': state})
    if self._get_progress() < 100:
        LOG.error('Aborting lease for %s due to incomplete transfer.', self._url)
        self._session.invoke_api(self._session.vim, 'HttpNfcLeaseAbort', self._lease)
    elif state == 'ready':
        LOG.debug('Releasing lease for %s.', self._url)
        self._session.invoke_api(self._session.vim, 'HttpNfcLeaseComplete', self._lease)
    else:
        LOG.debug('Lease for %(url)s is in state: %(state)s; no need to release.', {'url': self._url, 'state': state})