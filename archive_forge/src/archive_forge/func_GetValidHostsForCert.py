import re
import socket
import ssl
import boto
from boto.compat import six, http_client
def GetValidHostsForCert(cert):
    """Returns a list of valid host globs for an SSL certificate.

    Args:
      cert: A dictionary representing an SSL certificate.
    Returns:
      list: A list of valid host globs.
    """
    if 'subjectAltName' in cert:
        return [x[1] for x in cert['subjectAltName'] if x[0].lower() == 'dns']
    else:
        return [x[0][1] for x in cert['subject'] if x[0][0].lower() == 'commonname']