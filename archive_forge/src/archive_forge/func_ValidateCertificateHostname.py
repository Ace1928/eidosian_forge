import re
import socket
import ssl
import boto
from boto.compat import six, http_client
def ValidateCertificateHostname(cert, hostname):
    """Validates that a given hostname is valid for an SSL certificate.

    Args:
      cert: A dictionary representing an SSL certificate.
      hostname: The hostname to test.
    Returns:
      bool: Whether or not the hostname is valid for this certificate.
    """
    hosts = GetValidHostsForCert(cert)
    boto.log.debug('validating server certificate: hostname=%s, certificate hosts=%s', hostname, hosts)
    for host in hosts:
        host_re = host.replace('.', '\\.').replace('*', '[^.]*')
        if re.search('^%s$' % (host_re,), hostname, re.I):
            return True
    return False