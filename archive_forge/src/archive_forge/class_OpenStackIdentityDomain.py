import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityDomain:

    def __init__(self, id, name, enabled):
        self.id = id
        self.name = name
        self.enabled = enabled

    def __repr__(self):
        return '<OpenStackIdentityDomain id={}, name={}, enabled={}>'.format(self.id, self.name, self.enabled)