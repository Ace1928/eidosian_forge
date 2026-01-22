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
def _get_soap_url(self, scheme, host, port):
    """Returns the IPv4/v6 compatible SOAP URL for the given host."""
    if netutils.is_valid_ipv6(host):
        return '%s://[%s]:%d' % (scheme, host, port)
    return '%s://%s:%d' % (scheme, host, port)