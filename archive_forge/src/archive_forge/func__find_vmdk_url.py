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
@staticmethod
def _find_vmdk_url(lease_info, host, port):
    """Find the URL corresponding to a VMDK file in lease info."""
    url = None
    ssl_thumbprint = None
    for deviceUrl in lease_info.deviceUrl:
        if deviceUrl.disk:
            url = VmdkHandle._fix_esx_url(deviceUrl.url, host, port)
            ssl_thumbprint = deviceUrl.sslThumbprint
            break
    if not url:
        excep_msg = _('Could not retrieve VMDK URL from lease info.')
        LOG.error(excep_msg)
        raise exceptions.VimException(excep_msg)
    LOG.debug('Found VMDK URL: %s from lease info.', url)
    return (url, ssl_thumbprint)