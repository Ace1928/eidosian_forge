import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackAuthenticationContext:
    """
    An authentication token and related context.
    """

    def __init__(self, token, expiration=None, user=None, roles=None, urls=None):
        self.token = token
        self.expiration = expiration
        self.user = user
        self.roles = roles
        self.urls = urls